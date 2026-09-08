"""Generate the cached teacher signal for X-Token / hybrid KD (plan §7.2.1, §8.2).

`runner.py` never holds a live teacher — §8.2 runs it alone, caches, and unloads
it. This script is what produces that cache. It writes one `signal_kind`:
``topk_logits`` (top-K token ids + probabilities at each supervised answer
position), which is what both of the study's logit-based modes need
(`docs/New_Submission/experiment_protocol.md` "Three distillation modes":
sequence-level, X-Token P-KL, and the hybrid CE+X-Token — plain dense `token`
KD and `candidate` KD are not in that ladder, so the ~121 GB dense alternative
is never actually required here).

Teacher-forced, not free-generated: the teacher sees the gold prompt+answer
exactly as `distillation/train_student.py` builds it for the student, and only
the answer positions (the same span `masked_cross_entropy` supervises) are
cached. Same tokenizer family as the primary Qwen3.5-9B/0.8B pair means those
positions align 1:1 with the student's own labels with no span-alignment logic
needed; a cross-tokenizer pair would need `distillation.xtoken`'s span
machinery on top of this, which is out of scope for this script.

Stored at ``temperature: 1.0`` (plain softmax of the raw logits, top-K taken
after). That is a cache-defining choice, not a training one: which K tokens
survive truncation depends on the temperature used to rank them, so a different
loss-time `kd_temperature` re-scales the stored probabilities
(`xtoken.projected_kl_loss` divides and renormalizes) rather than requiring a
new cache, but a different *cache-time* temperature would select a different
top-K set and does require one — which is why `temperature` is a
`CACHE_KEY_FIELD` at all.

Usage::

    python distillation/build_teacher_cache.py --split train --top-k 4096 --limit 16
    python distillation/build_teacher_cache.py --split train --top-k 4096
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.cache import CacheKey, cache_directory, write_cache_key  # noqa: E402
from distillation.losses import IGNORE_INDEX  # noqa: E402
from distillation.teacher_cache_loader import GeneratedTextCache  # noqa: E402
from distillation.train_student import (  # noqa: E402
    PROMPT_SUFFIX, build_batch, build_batch_with_answers, build_image, load_rows)

DEFAULT_CACHE_ROOT = os.path.join(
    PROJECT_ROOT, "checkpoints_scratch", "teacher_cache")


def prompt_signature(processor) -> str:
    """Hashes the fully rendered chat template, not just `PROMPT_SUFFIX`.

    Hashing only the suffix text missed a real change once already: switching
    `enable_thinking` from Qwen3.5's default (an *open* `<think>\n` at the
    assistant turn) to closed changes every teacher-forced target — the answer
    is no longer being predicted as the model's own reasoning — but leaves the
    suffix string identical, so the old hash silently reused. Rendering the
    template with a placeholder question is what actually determines the
    tokens the teacher is forced over, so it is what must be hashed.
    """
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": f"placeholder\n{PROMPT_SUFFIX}"}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


def dataset_version() -> str:
    with open(os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2", "manifest.json"),
             encoding="utf-8") as handle:
        return json.load(handle)["version"]


def usable_cached_ids(directory: str, top_k: int) -> set:
    """Question ids already cached here whose file is intact.

    Resume exists because generation is a ~34-minute job that a server restart
    can interrupt (it did, at 9,152 of 15,278). Files are one per example, so
    the finished ones are reusable — but they are *verified* rather than
    trusted: a hard stop can leave the in-flight file truncated, and unflushed
    writes can lose more than just the last one. A corrupt file is deleted so
    the row is simply regenerated.

    Checked per file: both arrays load, agree on row count, and carry the K this
    run declares. A K mismatch means the file belongs to another configuration
    entirely, which the cache key should already have prevented — if it appears,
    the directory is not the one it claims to be, so this refuses rather than
    silently mixing two target sources.
    """
    import numpy as np

    usable, discarded = set(), 0
    for name in os.listdir(directory):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(directory, name)
        try:
            with np.load(path) as payload:
                ids, probs = payload["topk_ids"], payload["topk_probs"]
                if ids.shape != probs.shape or ids.ndim != 2 or ids.shape[0] == 0:
                    raise ValueError(f"malformed arrays {ids.shape} vs {probs.shape}")
                if ids.shape[1] != top_k:
                    raise SystemExit(
                        f"{path} holds top_k={ids.shape[1]} but this run declares "
                        f"{top_k}. That is a different target source in a directory "
                        f"whose key says otherwise — refusing to mix them.")
        except SystemExit:
            raise
        except Exception:
            os.remove(path)
            discarded += 1
            continue
        usable.add(name[:-len(".npz")])
    if discarded:
        print(f"  discarded {discarded} unreadable cache file(s) — they will be redone",
              flush=True)
    return usable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int,
                        help="First N rows only — use this to size or sanity-check "
                             "before committing to a full-split cache.")
    parser.add_argument("--out-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate every example even if an intact cache file "
                             "already exists. Default is to resume, which is what "
                             "makes an interrupted run cheap to finish.")
    parser.add_argument("--prefix-source", default="gold", choices=["gold", "teacher_generated"],
                        help="What answer text is forced as the teacher-forcing prefix "
                             "before caching top-K logits. 'gold' (default) is every "
                             "row D1-D8 use, and matches this script's behaviour before "
                             "this flag existed exactly — the key omits `prefix_source` "
                             "entirely in that case, so existing gold caches remain valid "
                             "and reusable. 'teacher_generated' is D9's strict label-access "
                             "rule (§8.1): the prefix is each row's own completion from "
                             "--generated-text-cache instead of row['answer'], which is "
                             "then not read at all.")
    parser.add_argument("--generated-text-cache",
                        help="Directory from build_teacher_generation_cache.py. Required "
                             "with --prefix-source teacher_generated.")
    args = parser.parse_args()

    if args.prefix_source == "teacher_generated" and not args.generated_text_cache:
        parser.error("--prefix-source teacher_generated needs --generated-text-cache "
                     "(a directory from build_teacher_generation_cache.py)")

    import torch
    from peft import LoraConfig  # noqa: F401  (import guard: fails loudly if peft is absent)
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.teacher)
    processor.tokenizer.padding_side = "right"

    # §9.3: "K and the tokenizer revisions are part of the frozen target
    # source — changing either invalidates the cache exactly as a precision
    # change does." `args.teacher` alone (e.g. "Qwen/Qwen3.5-9B") resolves to
    # whatever "main" is at run time, so it is not by itself a pinned
    # revision. `_commit_hash` only lands on config objects, not on the
    # tokenizer/processor themselves (checked directly: absent on both), so
    # AutoConfig is queried purely to read off what the loader resolved.
    teacher_revision = getattr(AutoConfig.from_pretrained(args.teacher), "_commit_hash", None)

    key = CacheKey({
        "dataset_version": dataset_version(),
        "split": args.split,
        "teacher_model": args.teacher,
        "teacher_revision": teacher_revision,
        "teacher_tokenizer_revision": teacher_revision,
        "precision": "bfloat16",
        "prompt_hash": prompt_signature(processor),
        "rgb_transform": "PIL RGB, processor default resize",
        "signal_kind": "topk_logits",
        "top_k": args.top_k,
        "temperature": 1.0,
        # None (not "gold") for the default case so this key's digest is
        # byte-for-byte identical to every cache built before this flag
        # existed — CacheKey.digest() reads every field with `.get(name)`, so
        # an explicit "gold" here would silently invalidate every existing
        # gold-prefix cache (the one D4/D5/D6/D7 already depend on) the next
        # time this script is run against it.
        "prefix_source": None if args.prefix_source == "gold" else args.prefix_source,
    })
    directory = cache_directory(args.out_root, key)
    os.makedirs(directory, exist_ok=True)
    write_cache_key(directory, key)

    generated_cache = None
    if args.prefix_source == "teacher_generated":
        # The generation cache's own key: same split/teacher/prompt as this
        # run declares, signal_kind="generated_text" — built independently by
        # build_teacher_generation_cache.py, verified here rather than trusted.
        generation_key = CacheKey({
            "dataset_version": dataset_version(),
            "split": args.split,
            "teacher_model": args.teacher,
            "teacher_revision": teacher_revision,
            "teacher_tokenizer_revision": teacher_revision,
            "precision": "bfloat16",
            "prompt_hash": prompt_signature(processor),
            "rgb_transform": "PIL RGB, processor default resize",
            "signal_kind": "generated_text",
        })
        generated_cache = GeneratedTextCache(args.generated_text_cache, generation_key)
        print(f"generated-text cache verified: {args.generated_text_cache} "
              f"({generation_key.digest()})", flush=True)
    print(f"cache directory: {directory}", flush=True)
    print(json.dumps(key.describe(), indent=2), flush=True)

    rows = load_rows(args.split, args.limit)
    requested = len(rows)
    resumed = 0
    if not args.overwrite:
        already = usable_cached_ids(directory, args.top_k)
        if already:
            rows = [row for row in rows if row["question_id"] not in already]
            resumed = requested - len(rows)
            print(f"resuming: {resumed} of {requested} already cached and intact; "
                  f"{len(rows)} to generate", flush=True)
    print(f"{len(rows)} {args.split} rows to do (of {requested}); "
          f"teacher={args.teacher}; top_k={args.top_k}", flush=True)

    # Loading a 9B teacher costs ~20 GB and a minute; there is nothing for it to
    # do when every row is already cached, and the summary below is still written
    # so a completed cache reports itself as complete.
    model = None
    if rows:
        model = AutoModelForImageTextToText.from_pretrained(
            args.teacher, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
    else:
        print("nothing to generate — cache is already complete for this key", flush=True)

    started = time.time()
    written = skipped = 0
    total_positions = 0
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        images = []
        kept = []
        for row in chunk:
            try:
                images.append(build_image(row, "rgb", "replicated"))
                kept.append(row)
            except Exception as error:                    # unreadable frame, bad row
                skipped += 1
                if skipped <= 3:
                    print(f"  skipped {row['question_id']}: {error}", flush=True)
        if not kept:
            continue

        if generated_cache is not None:
            answers = generated_cache.answers_for([row["question_id"] for row in kept])
            batch = build_batch_with_answers(processor, kept, images, answers)
        else:
            batch = build_batch(processor, kept, images)
        labels = batch.pop("labels")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = labels.to(model.device)

        with torch.inference_mode():
            logits = model(**batch).logits

        # Same causal shift `masked_cross_entropy` applies: position i's logits
        # predict token i+1, which is what labels[:, 1:] holds.
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != IGNORE_INDEX

        # Gather the handful of supervised positions (answers are 1-2 words,
        # §10.2) *before* the vocab-wide op, not after. Softmaxing the full
        # [B, L, 248320] sequence OOMs a 24 GB card outright even with only the
        # 9B teacher resident — masked_cross_entropy hit the identical shape of
        # bug (§10.1) and the fix is the same: narrow first, upcast second.
        flat_mask = mask.reshape(-1)
        selected_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))[flat_mask]
        probs = torch.softmax(selected_logits.float(), dim=-1)
        topk_probs, topk_ids = probs.topk(args.top_k, dim=-1)

        rows_per_example, length = shifted_labels.shape
        position_row = torch.arange(rows_per_example, device=labels.device
                                     ).repeat_interleave(length)[flat_mask]

        import numpy as np
        for row_index, row in enumerate(kept):
            example_mask = position_row == row_index
            n_positions = int(example_mask.sum())
            if n_positions == 0:
                skipped += 1
                continue
            payload = {
                "question_id": row["question_id"],
                "topk_ids": topk_ids[example_mask].to(torch.int32).cpu().numpy(),
                "topk_probs": topk_probs[example_mask].to(torch.float16).cpu().numpy(),
            }
            np.savez_compressed(
                os.path.join(directory, f"{row['question_id']}.npz"), **payload)
            written += 1
            total_positions += n_positions

        if written and written % args.log_every < args.batch_size:
            elapsed = time.time() - started
            print(f"  {written}/{len(rows)}  {written / elapsed:.2f} ex/s  "
                  f"avg positions/example {total_positions / written:.2f}", flush=True)

    elapsed = time.time() - started
    summary = {
        "teacher": args.teacher, "split": args.split, "top_k": args.top_k,
        "examples_written": written, "examples_skipped": skipped,
        "examples_resumed": resumed, "examples_in_split": requested,
        "examples_cached_total": resumed + written,
        "avg_positions_per_example": total_positions / written if written else None,
        "elapsed_minutes": round(elapsed / 60, 2),
        "examples_per_second": round(written / elapsed, 3) if elapsed else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "cache_key": key.describe(),
        "cache_digest": key.digest(),
    }
    with open(os.path.join(directory, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
