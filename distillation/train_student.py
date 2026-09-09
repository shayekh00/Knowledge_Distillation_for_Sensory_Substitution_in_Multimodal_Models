"""LoRA fine-tuning of the depth-only student (rows B3/B4, plan §8.4, §9.2).

The supervised baseline the whole paper is compared against. §4 of the protocol
is blunt about it: do not weaken CE by freezing modules KD is allowed to adapt or
by giving it fewer tuning trials, because a KD gain over a badly-trained CE
student proves nothing.

Correctness points carried over from `docs/New_Submission/implementation_audit.md`,
each of which the legacy path got wrong:

* **A3** — labels are masked to the **answer positions only**. The legacy collator
  masked padding alone, so CE trained the model to reproduce the question. Here
  the prompt is tokenized separately and its span is set to ``IGNORE_INDEX``.
* **A2** — depth is decoded with the official rotation via
  ``distillation.depth_input``, the same decoding the gold answers came from.
* **A1** — no augmentation is applied, and none is claimed. §8.4 runs the pilot
  with geometric augmentation off.
* **B3** — the loss is `masked_cross_entropy`, which averages over valid answer
  positions after a causal shift.

Peak VRAM and throughput are measured and written to the run manifest, because
Gate G4 asks for measured numbers rather than arithmetic.

Usage::

    python distillation/train_student.py --recipe B3 --epochs 1 --limit 2000
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

from distillation.depth_input import (  # noqa: E402
    decode_arkit_depth, decode_metric_depth, depth_to_student_input)
from distillation.epoch_loop import (  # noqa: E402
    EarlyStopper,
    generate_val_predictions,
    score_val_macro,
)
from distillation.losses import IGNORE_INDEX, masked_cross_entropy  # noqa: E402

PROMPT_SUFFIX = "Answer in one or two words. No explanation."


def load_rows(split: str, limit: int | None = None, csv_path: str | None = None) -> list:
    """`csv_path` overrides the frozen release CSV for this split — used only by
    ladder-external analyses (e.g. a leave-one-source-out split) that must read
    a derived, non-frozen file instead of `release/VQA-SUNRGBD-v2/rule_based/`,
    which G1 requires stay untouched."""
    path = csv_path or os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2",
                        "rule_based", f"{split}.csv")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def build_image(row, modality: str, representation: str, dataset: str = "sunrgbd"):
    """`dataset` picks the depth decoder (irrelevant for modality="rgb"):
    "sunrgbd" (default) is byte-for-byte the original behaviour
    (`decode_metric_depth`'s bit-rotation, A2). "arkitscenes" uses
    `decode_arkit_depth` instead — ARKitScenes' plain-millimetre depth
    encoding is not SUN-RGB-D's, so applying the bit-rotation to it would
    scramble every value rather than just decode it (arkitscenes_plan.md
    §6 Phase 5's prerequisite check, `tests/test_depth_input.py`)."""
    from PIL import Image
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    if modality == "rgb":
        return Image.open(os.path.join(dataset_dir, row["image_path"])).convert("RGB")
    decode = decode_arkit_depth if dataset == "arkitscenes" else decode_metric_depth
    metres = decode(os.path.join(dataset_dir, row["depth_path"]))
    return Image.fromarray(depth_to_student_input(metres, representation))


def build_batch_with_answers(processor, rows, images, answers):
    """A padded batch whose labels are masked to each row's own answer span,
    where the answer text for each row comes from `answers[row["question_id"]]`
    rather than always being `row["answer"]`.

    This is `build_batch`'s actual implementation, generalised over where the
    forced answer text comes from. D1-D8 always force the **gold** answer
    (`build_batch` below is exactly that special case); D9's strict label-access
    rule (`experiment_protocol.md` §8.1) instead forces the **teacher's own
    free-generated completion** for that row, with no gold text anywhere in the
    call — the two paths share every other detail (prompt rendering, thinking
    mode, padding, the mask-building arithmetic), and duplicating that would be
    exactly the kind of two-copies-that-drift risk this project has hit before
    with the teacher-forced/eval prompt mismatch this same function's
    docstring warns about below.

    Batch-1 training left the GPU latency-bound: measured 2.11 ex/s at batch 1
    against 12.84 ex/s at batch 4, with the data path accounting for 1.7% of a
    step. Batching is therefore where the time is, and getting the mask right
    across a padded batch is the whole difficulty.

    The prompt is encoded separately **with its image**, because the processor
    expands the image placeholder into many tokens and only it knows how many.
    Each row's prompt length is then its non-padding count, which is a valid
    prompt/answer boundary only under right padding — set by the caller.

    Batch size is a throughput knob only: `masked_cross_entropy` normalises per
    example, so batch 4 with accumulation 4 optimises the same objective as
    batch 1 with accumulation 16 (`test_ce_is_invariant_to_micro_batch_size`).
    A KD row that has to shrink its batch to fit the teacher cache therefore
    stays matched to CE under §4.
    """
    prompt_texts, full_texts = [], []
    for row in rows:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"{row['question']}\n{PROMPT_SUFFIX}"}]}]
        # enable_thinking=False is not optional here. Qwen3.5 defaults to an
        # *open* `<think>\n` at the assistant turn; without this the answer
        # text is teacher-forced as if it were the model's own reasoning, not
        # its final answer, which is a different and structurally incoherent
        # training target. `zero_shot_inference.py` already closes it
        # (`<think>\n\n</think>\n\n`) for every eval; this must match, or CE is
        # trained under a different prompt than the one it is scored under.
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                    tokenize=False, enable_thinking=False)
        prompt_texts.append(prompt_text)
        answer = answers[row["question_id"]]
        full_texts.append(prompt_text + str(answer) + processor.tokenizer.eos_token)

    prompt_inputs = processor(images=images, text=prompt_texts, padding=True,
                              return_tensors="pt")
    full_inputs = processor(images=images, text=full_texts, padding=True,
                            return_tensors="pt")

    labels = full_inputs["input_ids"].clone()
    for index, prompt_length in enumerate(prompt_inputs["attention_mask"].sum(dim=1).tolist()):
        labels[index, :prompt_length] = IGNORE_INDEX
    labels[full_inputs["attention_mask"] == 0] = IGNORE_INDEX

    batch = dict(full_inputs)
    batch["labels"] = labels
    return batch


def build_batch(processor, rows, images):
    """`build_batch_with_answers` forced to each row's own gold `answer` column
    — the D1-D8 case, and every call site before D9 existed. Kept as its own
    function (rather than inlining the dict comprehension at every call site)
    so nothing else has to change: existing behaviour is exactly preserved."""
    return build_batch_with_answers(
        processor, rows, images, {row["question_id"]: row["answer"] for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--recipe", default="B3")
    parser.add_argument("--modality", choices=["depth", "rgb"], default="depth")
    parser.add_argument("--dataset", choices=["sunrgbd", "arkitscenes"], default="sunrgbd",
                        help="Picks the depth decoder for --modality depth (irrelevant "
                             "for rgb): sunrgbd's bit-rotation vs. arkitscenes' plain-"
                             "millimetre PNGs (arkitscenes_plan.md §6 Phase 5). Does not "
                             "select which release CSV to read — use --train-csv/--val-csv "
                             "for that, same as the LOSO override.")
    parser.add_argument("--representation", choices=["replicated", "gradient"],
                        default="replicated")
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
    parser.add_argument("--limit", type=int, help="Train on the first N rows only.")
    parser.add_argument("--train-csv", help="Override the frozen release train.csv "
                        "with a derived CSV (e.g. a leave-one-source-out split). "
                        "Never points inside release/ — that directory is frozen (G1).")
    parser.add_argument("--val-csv", help="Same override for val.csv.")
    parser.add_argument("--val-release-dir", help="Directory holding the gold "
                        "{split}.csv that score_val_macro checks predictions "
                        "against. Required alongside --val-csv whenever that CSV is "
                        "a genuine subset (e.g. leave-one-source-out) — without it, "
                        "score_val_macro scores the subset's predictions against the "
                        "full frozen val.csv and every excluded row silently counts "
                        "as wrong (the exact bug this flag exists to prevent).")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Rows per forward/backward. 4 measured fastest here: "
                             "12.84 ex/s against 2.11 at batch 1, and 8 does not fit "
                             "without gradient checkpointing.")
    parser.add_argument("--effective-batch", type=int, default=16,
                        help="Rows per optimizer step. Accumulation is derived from "
                             "this and --batch-size, so changing the batch size does "
                             "not silently change the optimisation.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trades 1.85x throughput for memory this card does not "
                             "need (5.4 GB of 24 at batch 1). Off by default; required "
                             "only for batch sizes that would otherwise not fit.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--out", default="runs/pilot/B3_depth")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--confirmatory", dest="pilot", action="store_false",
                        help="Mark this run's resource_usage.json as CONFIRMATORY, not "
                             "PILOT. Matches evaluation/record_run.py's flag of the same "
                             "name and meaning (experiment_protocol.md §9.5): a PILOT "
                             "manifest may never enter a main or ablation table.")
    parser.set_defaults(pilot=True)
    args = parser.parse_args()

    if args.effective_batch % args.batch_size:
        parser.error(f"--effective-batch {args.effective_batch} is not a multiple of "
                     f"--batch-size {args.batch_size}; the optimiser step would not "
                     f"see the batch size it claims")
    accumulation = args.effective_batch // args.batch_size
    if args.val_csv and not args.val_release_dir:
        parser.error("--val-csv needs --val-release-dir pointing at the directory "
                     "that CSV lives in (its own val.csv, not the frozen release). "
                     "Without it, score_val_macro checks predictions for that subset "
                     "against the full frozen val.csv and every row the subset "
                     "excludes silently counts as wrong every epoch.")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows("train", args.limit, csv_path=args.train_csv)
    print(f"{len(rows)} training rows; recipe={args.recipe}; modality={args.modality}",
          flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    # Right padding puts every row's prompt at offset 0, which is what makes the
    # per-row prompt length in build_batch a usable prompt/answer boundary.
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")

    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    # Precomputed once: images do not change across epochs, and re-decoding
    # them every epoch's validation pass would be pure waste.
    val_rows = load_rows("val", args.val_limit, csv_path=args.val_csv)
    val_images = [build_image(row, args.modality, args.representation, args.dataset) for row in val_rows]
    stopper = EarlyStopper(max_epochs=args.epochs, patience=args.patience)

    order = list(range(len(rows)))
    random.shuffle(order)
    started = time.time()
    step = seen = skipped = 0
    running = 0.0
    history = []

    batches = logged_batches = 0
    next_log = args.log_every

    for epoch in range(args.epochs):
        for start in range(0, len(order), args.batch_size):
            chunk, images = [], []
            for index in order[start:start + args.batch_size]:
                row = rows[index]
                try:
                    images.append(build_image(row, args.modality, args.representation, args.dataset))
                    chunk.append(row)
                except Exception as error:               # unreadable frame, bad row
                    skipped += 1
                    if skipped <= 3:
                        print(f"  skipped {row['question_id']}: {error}", flush=True)
            if not chunk:
                continue

            batch = build_batch(processor, chunk, images)
            labels = batch.pop("labels")
            if (labels != IGNORE_INDEX).sum() == 0:
                skipped += len(chunk)    # nothing supervised; skip rather than crash
                continue

            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = masked_cross_entropy(outputs.logits, labels.to(model.device))
            (loss / accumulation).backward()
            running += float(loss.detach())
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
                # Batches no longer land on exact multiples of --log-every, so the
                # mean is over the batches actually since the last line.
                next_log = seen + args.log_every
                elapsed = time.time() - started
                rate = seen / elapsed
                mean_loss = running / logged_batches
                running, logged_batches = 0.0, 0
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  epoch {epoch} {seen}/{len(order)}  step {step}  "
                      f"loss {mean_loss:.4f}  {rate:.2f} ex/s  peak {peak:.2f} GB",
                      flush=True)
                history.append({"step": step, "examples": seen, "loss": mean_loss,
                                "examples_per_second": rate, "peak_vram_gb": peak})

        # End of epoch: the real task metric, not the training loss (§4 forbids
        # comparing CE and KD "by comparing differently scaled CE and KD
        # losses" — a stopping decision is exactly that comparison one level up).
        predictions = generate_val_predictions(model, processor, val_rows, val_images)
        score_kwargs = {"release_dir": args.val_release_dir} if args.val_release_dir else {}
        if args.dataset == "arkitscenes":
            score_kwargs["canonical_objects_dir"] = os.path.join(PROJECT_ROOT, "data", "vocab_arkit")
        val_macro = score_val_macro(predictions, split="val", **score_kwargs)
        epoch_dir = os.path.join(args.out, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(os.path.join(epoch_dir, "adapter"))
        should_stop = stopper.step(epoch, val_macro)
        print(f"  epoch {epoch} val macro {val_macro:.4f}  "
              f"best so far {stopper.best_macro:.4f} (epoch {stopper.best_epoch})"
              + ("  [stopping]" if should_stop else ""), flush=True)
        # Only the best-so-far epoch is ever read again (at the very end,
        # below) — training itself continues from the live in-memory model,
        # never by reloading a checkpoint from disk. So every superseded
        # epoch's adapter is deleted immediately rather than left until the
        # run finishes.
        for name in os.listdir(args.out):
            if name.startswith("epoch_") and name != f"epoch_{stopper.best_epoch}":
                shutil.rmtree(os.path.join(args.out, name))
        if should_stop:
            break

    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1e9
    os.makedirs(args.out, exist_ok=True)
    # Early stopping means keeping the best epoch, not whichever one triggered
    # the stop — those are usually different epochs (patience is the gap
    # between them by construction). It is the only epoch_* directory left on
    # disk by this point, so promote it by moving rather than copying.
    best_epoch_dir = os.path.join(args.out, f"epoch_{stopper.best_epoch}")
    final_adapter = os.path.join(args.out, "adapter")
    if os.path.exists(final_adapter):
        shutil.rmtree(final_adapter)
    shutil.move(os.path.join(best_epoch_dir, "adapter"), final_adapter)
    shutil.rmtree(best_epoch_dir)

    resources = {
        "recipe": args.recipe, "model": args.model, "modality": args.modality,
        "seed": args.seed, "epochs": args.epochs, "examples_seen": seen,
        "examples_skipped": skipped, "optimizer_steps": step,
        "learning_rate": args.learning_rate, "accumulation": accumulation,
        "batch_size": args.batch_size, "effective_batch": args.effective_batch,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_rank": args.lora_rank,
        "elapsed_minutes": round(elapsed / 60, 2),
        "examples_per_second": round(seen / elapsed, 3) if elapsed else None,
        "peak_vram_gb": round(peak, 2),
        "gpu": torch.cuda.get_device_name(0),
        "pilot": args.pilot,
        "projected_minutes_full_train_epoch": (
            round((15278 / (seen / elapsed)) / 60, 1) if seen and elapsed else None),
        "max_epochs": args.epochs, "patience": args.patience,
        "epochs_run": epoch + 1, "best_epoch": stopper.best_epoch,
        "best_val_macro": stopper.best_macro,
        "stopped_on": ("max_epochs" if epoch + 1 >= args.epochs
                      and stopper.epochs_since_improvement < args.patience
                      else "patience"),
    }
    with open(os.path.join(args.out, "resource_usage.json"), "w", encoding="utf-8") as handle:
        json.dump({"resources": resources, "history": history, "val_history": stopper.history}, handle, indent=2)

    print(json.dumps(resources, indent=2), flush=True)
    print(f"adapter saved to {args.out}/adapter", flush=True)


if __name__ == "__main__":
    main()
