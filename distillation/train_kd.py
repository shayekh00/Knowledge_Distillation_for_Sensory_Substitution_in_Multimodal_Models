"""Train a student against a cached teacher signal (plan §8.2, protocol §9.3).

The KD counterpart to `train_student.py`. Same data path, same batch machinery,
same per-example loss normalisation — the difference is that the objective comes
from `runner.compose_loss` under a `RecipeConfig` rather than being CE by name,
so a row is *selected* (`--recipe X2`) rather than coded.

**No teacher is ever loaded here.** §8.2 runs the teacher alone, caches, and
unloads it; this script reads `.npz` rows off disk. That is what keeps a 24 GB
card viable: the student's ~14.6 GB peak never has to coexist with the teacher's
~20.6 GB. Attempting both at once OOMs, measured.

Every loss component is written to `training_metrics.csv` per logging interval,
not just the total. `compose_loss`'s own docstring gives the reason: the previous
submission shipped a KD term that was effectively off, and a component log is
what makes that diagnosable rather than invisible.

Usage::

    python distillation/train_kd.py --recipe X2 --cache <dir> --limit 64
    python distillation/train_kd.py --recipe X2 --cache <dir> --confirmatory
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.build_teacher_cache import dataset_version, prompt_signature  # noqa: E402
from distillation.cache import CacheKey  # noqa: E402
from distillation.epoch_loop import (  # noqa: E402
    EarlyStopper,
    generate_val_predictions,
    score_val_macro,
)
from distillation.losses import IGNORE_INDEX  # noqa: E402
from distillation.runner import TeacherSignals, compose_loss, recipe_library  # noqa: E402
from distillation.teacher_cache_loader import (  # noqa: E402
    GeneratedTextCache,
    QwenAdapter,
    TeacherCache,
    assert_rows_align,
)
from distillation.train_student import (  # noqa: E402
    build_batch, build_batch_with_answers, build_image, load_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", default="X2",
                        help="Row id from runner.recipe_library (X2 = CE + X-Token).")
    parser.add_argument("--cache",
                        help="Top-K logits cache from build_teacher_cache.py. Required "
                             "for any recipe with a logit KD term; a pure stage-F row "
                             "(D0) needs only --feature-cache.")
    parser.add_argument("--feature-cache",
                        help="Pooled-features cache from build_feature_cache.py. "
                             "Required for any recipe with a feature objective "
                             "(D0, D3, D5, D8).")
    parser.add_argument("--generated-text-cache",
                        help="generated_text cache from build_teacher_generation_cache.py. "
                             "Required for D9 only (§8.1): its --cache must itself be a "
                             "teacher_generated-prefix topk_logits cache "
                             "(build_teacher_cache.py --prefix-source teacher_generated), "
                             "and this is the matching generated_text cache D9's own "
                             "training batches are built against instead of gold.")
    parser.add_argument("--stage", default="auto", choices=["auto", "F", "P", "S2"],
                        help="Which half of a two-stage row to run. 'auto' runs the "
                             "recipe exactly as declared. 'F' runs its derived "
                             "feature-alignment-only stage; 'P' runs feature alignment "
                             "plus a small raw KD term (§8.2; D6 only); 'S2' runs its "
                             "answer stage from --parent-checkpoint (protocol §13.1 "
                             "point 5, resolved 2026-09-07 in favour of two stages).")
    parser.add_argument("--parent-checkpoint",
                        help="A stage-F or stage-P run directory whose adapter is "
                             "merged into the base weights before this stage's LoRA is "
                             "attached. Required with --stage S2 on a two-stage row.")
    parser.add_argument("--p-lambda-kd", type=float, default=0.1,
                        help="Weight of stage P's raw KD term (§8.2's \"small raw KD "
                             "term\", D6 only). Not fixed by the protocol; default is "
                             "an explicit, overridable author choice.")
    parser.add_argument("--negative-bank", type=int, default=255,
                        help="Negatives per contrastive step, drawn from that many "
                             "distinct scenes (audit B4: 255 + 1 positive).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-9B",
                        help="Named only to rebuild the cache key for verification; "
                             "the teacher itself is never loaded.")
    parser.add_argument("--modality", default="depth", choices=["depth", "rgb"])
    parser.add_argument("--representation", default="replicated",
                        choices=["replicated", "gradient"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max epochs; stops earlier under --patience "
                             "(protocol §7.3).")
    parser.add_argument("--patience", type=int, default=2,
                        help="Stop after this many consecutive epochs with no "
                             "val macro improvement (protocol §7.3).")
    parser.add_argument("--val-limit", type=int,
                        help="First N val rows for per-epoch scoring (smoke test "
                             "only — the real stopping decision needs the full "
                             "split, or it is not comparable across runs).")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--lambda-kd", type=float,
                        help="Override the recipe's KD weight.")
    parser.add_argument("--kd-temperature", type=float,
                        help="Override the recipe's KD temperature. The cache is stored "
                             "at temperature 1.0; this rescales it at loss time.")
    parser.add_argument("--limit", type=int, help="First N rows only (smoke test).")
    parser.add_argument("--out", default="runs/kd/X2_depth")
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--confirmatory", dest="pilot", action="store_false")
    parser.set_defaults(pilot=True)
    args = parser.parse_args()

    if args.effective_batch % args.batch_size:
        parser.error(f"--effective-batch {args.effective_batch} is not a multiple of "
                     f"--batch-size {args.batch_size}")
    accumulation = args.effective_batch // args.batch_size

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    library = recipe_library(top_k=args.top_k)
    if args.recipe not in library:
        parser.error(f"unknown recipe {args.recipe!r}; have {sorted(library)}")
    config = library[args.recipe]
    config.seed = args.seed
    config.lora_rank = args.lora_rank
    config.pilot = args.pilot
    if args.lambda_kd is not None:
        config.lambda_kd = args.lambda_kd
    if args.kd_temperature is not None:
        config.kd_temperature = args.kd_temperature

    # Resolve which half of the row this invocation is, before anything is
    # loaded. A two-stage row run as a single pass is refused by the surface
    # check below, so this is the only way to run D3/D5/D6/D8 at all.
    if args.stage == "F":
        config = config.stage_one()
    elif args.stage == "P":
        config = config.stage_one_p(p_lambda_kd=args.p_lambda_kd)
    elif args.stage == "S2":
        if config.is_two_stage() and not args.parent_checkpoint:
            parser.error(
                f"--stage S2 on two-stage row {args.recipe} needs --parent-checkpoint "
                f"(its stage-F run directory). Without the aligned vision weights this "
                f"is just the row's answer objective with nothing distilled into it.")
        if args.parent_checkpoint:
            config = config.stage_two(args.parent_checkpoint)
    elif config.is_two_stage():
        parser.error(
            f"recipe {args.recipe} is a two-stage row (feature_objective="
            f"{config.feature_objective!r}). Run it as --stage F then --stage S2 "
            f"--parent-checkpoint <F run dir>, or use "
            f"distillation/train_two_stage.py which chains both. Running it as one "
            f"pass would leave the feature term unable to reach any student "
            f"parameter (experiment_protocol.md §13.1 point 5).")
    print(f"stage: {config.stage} (--stage {args.stage})", flush=True)

    # Before anything is loaded: refuse a row whose objective cannot reach a
    # parameter (see the method's docstring — this is a measured failure mode,
    # not a hypothetical one).
    config.assert_trainable_surface_can_learn()

    needs_logits = config.kd_objective not in ("none", "sequence")
    if needs_logits and not args.cache:
        parser.error(f"recipe {args.recipe} has a logit KD term and needs --cache")
    if config.feature_objective != "none" and not args.feature_cache:
        parser.error(
            f"recipe {args.recipe} declares feature_objective="
            f"{config.feature_objective!r} and needs --feature-cache "
            f"(build_feature_cache.py). Running it without one would leave the "
            f"feature term unsatisfiable.")
    if not needs_logits and args.cache:
        print("note: --cache ignored; this recipe has no logit KD term", flush=True)
    if config.kd_objective == "candidate":
        parser.error(
            f"recipe {args.recipe} needs cached candidate scores, not top-K logits.")

    # §8.1's strict label-access rule: D9 is the only row whose KD prefix is
    # not the gold answer, so it is the only row needing the matching
    # teacher_generated topk_logits cache and its generated_text counterpart.
    strict_label_access = args.recipe == "D9"
    if strict_label_access and not args.generated_text_cache:
        parser.error("recipe D9 needs --generated-text-cache "
                     "(build_teacher_generation_cache.py's output directory)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows("train", args.limit)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"

    # The key must match what build_teacher_cache.py declared, or the cache is
    # not the one this configuration means to train against.
    from transformers import AutoConfig
    teacher_revision = getattr(AutoConfig.from_pretrained(args.teacher), "_commit_hash", None)
    student_revision = getattr(AutoConfig.from_pretrained(args.model), "_commit_hash", None)
    key = CacheKey({
        "dataset_version": dataset_version(), "split": "train",
        "teacher_model": args.teacher, "teacher_revision": teacher_revision,
        "teacher_tokenizer_revision": teacher_revision, "precision": "bfloat16",
        "prompt_hash": prompt_signature(processor),
        "rgb_transform": "PIL RGB, processor default resize",
        "signal_kind": "topk_logits", "top_k": args.top_k, "temperature": 1.0,
        # None for every row but D9 — see build_teacher_cache.py's identical
        # comment: an explicit "gold" here would change the digest for D1-D8
        # and invalidate every cache already verified against them.
        "prefix_source": "teacher_generated" if strict_label_access else None,
    })
    cache = None
    if needs_logits:
        cache = TeacherCache(args.cache, key)
        print(f"logits cache verified: {args.cache} ({key.digest()})", flush=True)

    generated_cache = None
    if strict_label_access:
        generation_key = CacheKey({
            "dataset_version": dataset_version(), "split": "train",
            "teacher_model": args.teacher, "teacher_revision": teacher_revision,
            "teacher_tokenizer_revision": teacher_revision, "precision": "bfloat16",
            "prompt_hash": prompt_signature(processor),
            "rgb_transform": "PIL RGB, processor default resize",
            "signal_kind": "generated_text",
        })
        generated_cache = GeneratedTextCache(args.generated_text_cache, generation_key)
        print(f"generated-text cache verified: {args.generated_text_cache} "
              f"({generation_key.digest()})", flush=True)
        # §8.1: "the gold answer column must be removed before the
        # training/cache interface, and the run succeeds with them absent" —
        # not merely unused. Stripped once here rather than trusted to stay
        # unread through the rest of the loop.
        rows = [{k: v for k, v in row.items() if k != "answer"} for row in rows]

    feature_cache = None
    if config.feature_objective != "none":
        from distillation.build_feature_cache import CROP_AGGREGATION, FEATURE_LAYER
        from distillation.teacher_cache_loader import FeatureCache
        feature_key = CacheKey({
            "dataset_version": dataset_version(), "split": "train",
            "teacher_model": args.teacher, "teacher_revision": teacher_revision,
            "processor_revision": teacher_revision, "precision": "bfloat16",
            "prompt_hash": None,
            "rgb_transform": "PIL RGB, processor default resize",
            "signal_kind": "pooled_features", "feature_layer": FEATURE_LAYER,
            "crop_aggregation": CROP_AGGREGATION,
        })
        feature_cache = FeatureCache(args.feature_cache, feature_key)
        print(f"feature cache verified: {args.feature_cache} ({feature_key.digest()}); "
              f"{feature_cache.features.size(0)} images, dim {feature_cache.feature_dim}",
              flush=True)
    print(f"{len(rows)} training rows; recipe={args.recipe}; "
          f"objective={config.kd_objective}; modality={args.modality}", flush=True)
    print(config.to_yaml_like(), flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    if config.parent_checkpoint:
        # Stage one's LoRA is *merged into the base weights* rather than kept as
        # a live adapter. Two reasons: §8.2 freezes the vision encoder in S2, and
        # merged weights are plain frozen parameters, so the freeze is structural
        # rather than a flag someone can forget; and PEFT would otherwise have to
        # hold two adapters with only one of them trainable, which is the kind of
        # state that silently trains the wrong surface.
        from peft import PeftModel
        parent_adapter = os.path.join(config.parent_checkpoint, "adapter")
        if not os.path.isdir(parent_adapter):
            parser.error(f"--parent-checkpoint {config.parent_checkpoint} has no "
                         f"adapter/ directory to merge")
        model = PeftModel.from_pretrained(model, parent_adapter)
        model = model.merge_and_unload()
        print(f"merged stage-one adapter from {parent_adapter}; vision is now frozen "
              f"base weights", flush=True)
    # The recipe declares which surface it trains; honour it rather than
    # hardcoding one. X2's ("language_attention",) happens to equal the q/k/v/o
    # projections, so this is a no-op for the current ladder — but a recipe
    # declaring anything else would otherwise train the wrong parameters and
    # report itself as that recipe, which is precisely the silent drift the run
    # id and cache key machinery exists to prevent elsewhere.
    #
    # Each surface is a *regex over fully-qualified module names*, not a list of
    # suffixes. PEFT matches a list entry with `endswith`, and the vision blocks'
    # attention output is named `proj` — which `o_proj`, `q_proj`, `k_proj` and
    # `v_proj` all end with. A list containing "proj" would therefore silently
    # wrap the language model's attention too, and the row would report itself as
    # vision-only while training language parameters. Anchored regexes make the
    # surface mean exactly what it says.
    module_targets = {
        "language_attention": r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj)$",
        "vision_attention": r".*visual\.blocks\.\d+\.attn\.(qkv|proj)$",
        # Anchored on `visual.merger` specifically, not on the leaf name alone:
        # every one of the 12 vision blocks' MLPs is *also* named
        # `mlp.linear_fc1` / `mlp.linear_fc2` (checked directly on the real
        # model). A suffix-style match on "linear_fc" would have silently
        # wrapped all 12 block MLPs alongside the one merger the author asked
        # for — the same class of collision `o_proj`/`proj` was for vision vs
        # language attention.
        "vision_merger": r".*visual\.merger\.(linear_fc1|linear_fc2)$",
    }
    unsupported = [m for m in config.trainable_modules if m not in module_targets]
    if unsupported:
        parser.error(
            f"recipe {args.recipe} declares trainable_modules {unsupported}, which this "
            f"adapter cannot express as LoRA targets. Implement them before running the "
            f"row rather than silently training {sorted(module_targets)}.")
    targets = "|".join(f"(?:{module_targets[m]})" for m in sorted(set(config.trainable_modules)))
    print(f"trainable surface: {config.trainable_modules} -> LoRA target regex {targets}",
          flush=True)
    lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()
    alignment_head = None
    if feature_cache is not None:
        from distillation.features import AlignmentHead
        student_dim = model.config.vision_config.out_hidden_size
        alignment_head = AlignmentHead(student_dim, feature_cache.feature_dim).to(
            model.device, dtype=torch.float32)
        print(f"alignment head: {student_dim} -> {feature_cache.feature_dim} "
              f"({sum(p.numel() for p in alignment_head.parameters())} params, "
              f"stage-F only, not part of the deployable student)", flush=True)
    adapter = QwenAdapter(model, alignment_head=alignment_head)

    mapping = None
    gold_teacher_lookup = None
    if config.kd_objective == "xtoken":
        from distillation.xtoken import (
            build_student_to_teacher_lookup,
            format_mapping_report,
            load_or_build_mapping,
        )
        from transformers import AutoTokenizer
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
        mapping = load_or_build_mapping(
            processor.tokenizer, teacher_tokenizer,
            cache_dir=os.path.join(PROJECT_ROOT, "checkpoints_scratch", "xtoken_mappings"),
            student_revision=student_revision or "unknown",
            teacher_revision=teacher_revision or "unknown")
        print(format_mapping_report(mapping), flush=True)
        if config.use_loca:
            # Depends only on the mapping, never on a batch, so it is built
            # once here and reused for every training step rather than
            # rebuilt from the sparse index arrays each forward pass. Moved to
            # the model's device now, not lazily on first use: gold_student_ids
            # (derived from labels, already on model.device) would otherwise
            # index a CPU tensor with a CUDA tensor and crash on step one.
            gold_teacher_lookup = build_student_to_teacher_lookup(mapping).to(model.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if alignment_head is not None:
        # Without this the head never updates and the feature loss is measured
        # against a fixed random projection of the student.
        trainable += list(alignment_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    # Seeded separately from the data order so the negative bank is reproducible
    # for a given seed without depending on how many batches preceded it.
    bank_generator = torch.Generator().manual_seed(args.seed)

    # The vision-alignment stage *is* scored on the full val split, like every
    # other row. An earlier version of this script skipped it, on the reasoning
    # that a stage-F row never trains the language model and so "has no task
    # metric" — that reasoning was wrong, and the 0.06% that appeared to support
    # it was an artifact of `--val-limit 8`: `score_val_macro` loads the whole
    # 1,720-row gold split regardless of how many predictions it is handed, so 8
    # predictions scored as 8/1720. The same artifact produced a 7.1% reading in
    # September and is now on its third appearance.
    #
    # The number is real and interpretable: it is the *pretrained* language model
    # reading the newly aligned vision, directly comparable to B1's 36.18%
    # zero-shot depth baseline, which used the same language model on unaligned
    # vision. Recording it per epoch is what lets the alignment budget be chosen
    # from a curve instead of guessed.
    #
    # It still does not *select* the checkpoint. What it measures is how legible
    # the shifted vision is to a language model that has not adapted to it, and
    # that can diverge from how good a starting point the shift is once S2 does
    # adapt. So stage F runs its declared budget and keeps its last epoch, while
    # reporting which epoch read best — evidence for the next budget choice,
    # not an automatic decision.
    # Stage P (D6) is a stage-one variant, not a variable-budget answer stage,
    # so it gets exactly the same fixed-budget/kept-last/no-early-stopping
    # treatment as stage F rather than a third code path.
    stage_f = config.stage in ("F", "P")
    val_rows = load_rows("val", args.val_limit)
    val_images = [build_image(row, args.modality, args.representation)
                  for row in val_rows]
    stopper = EarlyStopper(max_epochs=args.epochs, patience=args.patience)
    if stage_f:
        print(f"vision alignment (stage {config.stage}): {args.epochs} epoch(s), "
              f"full-val scored each epoch for the record, last epoch kept. Reference: "
              f"B1 zero-shot "
              f"depth = 36.18% with this same language model on unaligned vision.",
              flush=True)

    order = list(range(len(rows)))
    random.shuffle(order)
    started = time.time()
    step = seen = skipped = 0
    batches = logged_batches = 0
    next_log = args.log_every
    running: dict[str, float] = {}
    history = []

    for epoch in range(args.epochs):
        for start in range(0, len(order), args.batch_size):
            chunk, images = [], []
            for index in order[start:start + args.batch_size]:
                row = rows[index]
                if cache is not None and not cache.has(row["question_id"]):
                    skipped += 1                          # not in the cached split
                    continue
                try:
                    images.append(build_image(row, args.modality, args.representation))
                    chunk.append(row)
                except Exception as error:                # unreadable frame, bad row
                    skipped += 1
                    if skipped <= 3:
                        print(f"  skipped {row['question_id']}: {error}", flush=True)
            if not chunk:
                continue

            if generated_cache is not None:
                answers = generated_cache.answers_for([row["question_id"] for row in chunk])
                batch = build_batch_with_answers(processor, chunk, images, answers)
            else:
                batch = build_batch(processor, chunk, images)
            labels = batch.pop("labels")
            if (labels != IGNORE_INDEX).sum() == 0:
                skipped += len(chunk)
                continue

            batch = {key_: value.to(model.device) for key_, value in batch.items()}
            batch["labels"] = labels.to(model.device)

            if cache is not None:
                teacher = cache.signals_for([row["question_id"] for row in chunk],
                                            device=model.device)
                # The student's supervised positions and the cache's rows must
                # line up per example, not merely in total — see the loader's
                # docstring.
                assert_rows_align(teacher, batch["labels"])
            else:
                teacher = TeacherSignals()
            if feature_cache is not None:
                features = feature_cache.signals_for(
                    chunk, bank_size=args.negative_bank,
                    generator=bank_generator, device=model.device)
                teacher.features = features.features
                teacher.negative_bank = features.negative_bank
                teacher.metadata = {**teacher.metadata, **features.metadata}

            loss, components = compose_loss(config, adapter, batch, teacher,
                                            xtoken_mapping=mapping,
                                            gold_teacher_lookup=gold_teacher_lookup)
            (loss / accumulation).backward()

            for name, value in components.items():
                running[name] = running.get(name, 0.0) + value
            seen += len(chunk)
            batches += 1
            logged_batches += 1

            if batches % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            if seen >= next_log:
                next_log = seen + args.log_every
                elapsed = time.time() - started
                rate = seen / elapsed
                means = {name: total / logged_batches for name, total in running.items()}
                running, logged_batches = {}, 0
                peak = torch.cuda.max_memory_allocated() / 1e9
                parts = "  ".join(f"{name} {value:.4f}" for name, value in sorted(means.items()))
                print(f"  epoch {epoch} {seen}/{len(order)}  step {step}  {parts}  "
                      f"{rate:.2f} ex/s  peak {peak:.2f} GB", flush=True)
                history.append({"step": step, "examples": seen, **means,
                                "examples_per_second": rate, "peak_vram_gb": peak})

        # Real task metric, not the training loss — §4 forbids comparing CE and
        # KD "by comparing differently scaled CE and KD losses," and a stopping
        # decision on raw loss is exactly that one level up.
        predictions = generate_val_predictions(model, processor, val_rows, val_images)
        val_macro = score_val_macro(predictions, split="val")
        epoch_dir = os.path.join(args.out, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(os.path.join(epoch_dir, "adapter"))
        if alignment_head is not None:
            # Saved for stage chaining and inspection, not for deployment: §6's
            # deployable student never runs this head. Kept beside the adapter
            # so a stage-F checkpoint is self-describing.
            torch.save({"state_dict": alignment_head.state_dict(),
                        "student_dim": alignment_head.student_dim,
                        "teacher_dim": alignment_head.teacher_dim},
                       os.path.join(epoch_dir, "alignment_head.pt"))
        # Tracked in both cases; only acted on outside stage F.
        patience_reached = stopper.step(epoch, val_macro)
        if stage_f:
            keep_epoch = epoch                      # declared budget: keep last
            should_stop = (epoch + 1) >= args.epochs
            print(f"  epoch {epoch} feature loss "
                  f"{components.get('feature', float('nan')):.4f}  "
                  f"val macro {val_macro:.4f} (frozen LM reading aligned vision; "
                  f"best-reading epoch so far {stopper.best_epoch} at "
                  f"{stopper.best_macro:.4f})  [keeping last]"
                  + ("  [done]" if should_stop else ""), flush=True)
        else:
            should_stop = patience_reached
            keep_epoch = stopper.best_epoch
            print(f"  epoch {epoch} val macro {val_macro:.4f}  "
                  f"best so far {stopper.best_macro:.4f} (epoch {stopper.best_epoch})"
                  + ("  [stopping]" if should_stop else ""), flush=True)
        # Only the best-so-far epoch is ever read again (at the very end,
        # below) — training itself continues from the live in-memory model,
        # never by reloading a checkpoint from disk. So every superseded
        # epoch's adapter is deleted immediately rather than left until the
        # run finishes.
        for name in os.listdir(args.out):
            if name.startswith("epoch_") and name != f"epoch_{keep_epoch}":
                shutil.rmtree(os.path.join(args.out, name))
        if should_stop:
            break

    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1e9
    os.makedirs(args.out, exist_ok=True)
    # Keep the best epoch, not whichever one triggered the stop — those are
    # usually different epochs by exactly `patience`. It is the only epoch_*
    # directory left on disk by this point, so promote it by moving rather
    # than copying.
    best_epoch_dir = os.path.join(args.out, f"epoch_{keep_epoch}")
    final_adapter = os.path.join(args.out, "adapter")
    if os.path.exists(final_adapter):
        shutil.rmtree(final_adapter)
    shutil.move(os.path.join(best_epoch_dir, "adapter"), final_adapter)
    # The head lives beside the adapter inside the epoch directory, so it has to
    # be promoted too — the rmtree below would otherwise delete the only copy of
    # the stage-F projection this checkpoint's features were aligned through.
    best_head = os.path.join(best_epoch_dir, "alignment_head.pt")
    if os.path.isfile(best_head):
        shutil.move(best_head, os.path.join(args.out, "alignment_head.pt"))
    shutil.rmtree(best_epoch_dir)

    if history:
        with open(os.path.join(args.out, "training_metrics.csv"), "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(
                {name for entry in history for name in entry}))
            writer.writeheader()
            writer.writerows(history)

    resources = {
        "recipe": args.recipe, "model": args.model, "teacher": args.teacher,
        "modality": args.modality, "seed": args.seed, "epochs": args.epochs,
        "examples_seen": seen, "examples_skipped": skipped, "optimizer_steps": step,
        "learning_rate": args.learning_rate, "accumulation": accumulation,
        "batch_size": args.batch_size, "effective_batch": args.effective_batch,
        "gradient_checkpointing": False, "lora_rank": args.lora_rank,
        "elapsed_minutes": round(elapsed / 60, 2),
        "examples_per_second": round(seen / elapsed, 3) if elapsed else None,
        "peak_vram_gb": round(peak, 2), "gpu": torch.cuda.get_device_name(0),
        "pilot": args.pilot, "cache_digest": key.digest(),
        "distillation_mode": config.kd_objective,
        "projected_minutes_full_train_epoch": (
            round((15278 / (seen / elapsed)) / 60, 1) if seen and elapsed else None),
        "max_epochs": args.epochs,
        # Stage F runs a declared budget rather than an early-stopped one, so it
        # reports no patience instead of a CLI default it never consulted.
        "patience": None if stage_f else args.patience,
        "epochs_run": epoch + 1,
        "best_epoch": keep_epoch,
        "best_val_macro": stopper.best_macro,
        # Stage F only: which epoch the pretrained LM read best, recorded as
        # evidence for choosing --alignment-epochs. Not what selected the
        # checkpoint (see the comment above the epoch loop).
        "best_reading_epoch": stopper.best_epoch if stage_f else None,
        "stopped_on": ("fixed_budget" if stage_f else
                       ("max_epochs" if epoch + 1 >= args.epochs
                        and stopper.epochs_since_improvement < args.patience
                        else "patience")),
        "stage": config.stage,
        "parent_checkpoint": config.parent_checkpoint,
    }
    with open(os.path.join(args.out, "resource_usage.json"), "w", encoding="utf-8") as handle:
        json.dump({"resources": resources, "recipe_config": config.resolved(),
                   "history": history,
                   "val_history": stopper.history}, handle, indent=2)
    with open(os.path.join(args.out, "recipe_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config.resolved(), handle, indent=2)
    print(json.dumps(resources, indent=2), flush=True)
    print(f"adapter saved to {os.path.join(args.out, 'adapter')}", flush=True)


if __name__ == "__main__":
    main()
