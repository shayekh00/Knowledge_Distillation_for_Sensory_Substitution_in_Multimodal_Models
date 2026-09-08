"""Generate the teacher's own free completions, with no gold answer anywhere
in the prompt (experiment_protocol.md §8.1, D9's strict label-access rule).

This is the artifact `build_teacher_cache.py` cannot produce: that script is
teacher-**forced** — the teacher sees the gold prompt+answer exactly as
`train_student.py` builds it for the student, which is why every row D1-D8
distill from (however little CE they use) still leaks the gold answer into the
KD target as a teacher-forcing prefix. D9 is the one row in the matrix that may
not do that at all. This script produces the substitute: the teacher's own
greedy completion for each question, generated with the *same* prompt the
student is evaluated under and nothing else — no answer, gold or otherwise,
anywhere in its input.

The output of this script (`signal_kind: generated_text`) is not itself what
D9 trains against. It is the *input* to a second pass:
`build_teacher_cache.py --prefix-source teacher_generated --generated-text-cache
<this script's output directory>`, which forces each row's cached completion
(instead of `row["answer"]`) and caches the teacher's own top-K logits over it
— the same `topk_logits` shape every KD row already consumes, just built from a
different prefix source. That two-step split exists so the (much cheaper)
free-generation pass can be redone or inspected independently of the
(much more expensive) logits pass, and so a mistake in one is not a silent
mistake in the other.

Greedy decoding, the frozen `terse` prompt (`experiment_protocol.md` §8.3),
`enable_thinking=False` — identical contract to every other measured number in
this study (B1/B2, `zero_shot_inference.py`), because a teacher target
generated under a different prompt or decoding policy than what the student is
scored under is not a fair substitute for gold.

Usage::

    python distillation/build_teacher_generation_cache.py --split train --limit 16
    python distillation/build_teacher_generation_cache.py --split train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.build_teacher_cache import dataset_version, prompt_signature  # noqa: E402
from distillation.cache import CacheKey, cache_directory, write_cache_key  # noqa: E402
from distillation.train_student import PROMPT_SUFFIX, build_image, load_rows  # noqa: E402

DEFAULT_CACHE_ROOT = os.path.join(PROJECT_ROOT, "checkpoints_scratch", "teacher_cache")
MAX_NEW_TOKENS = 16
COMPLETIONS_FILENAME = "completions.json"


def build_cache_key(processor, split: str, teacher: str, teacher_revision) -> CacheKey:
    return CacheKey({
        "dataset_version": dataset_version(),
        "split": split,
        "teacher_model": teacher,
        "teacher_revision": teacher_revision,
        "teacher_tokenizer_revision": teacher_revision,
        "precision": "bfloat16",
        "prompt_hash": prompt_signature(processor),
        "rgb_transform": "PIL RGB, processor default resize",
        "signal_kind": "generated_text",
    })


def load_partial(directory: str) -> dict:
    """Resume support: a completed-so-far `{question_id: text}` table, saved
    incrementally so an interrupted run (this project has already hit a
    server-restart mid-run once — experiment_protocol.md's restart-safety row,
    2026-09-08) loses only what was generated since the last save, not the
    whole pass."""
    path = os.path.join(directory, COMPLETIONS_FILENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_partial(directory: str, completions: dict) -> None:
    path = os.path.join(directory, COMPLETIONS_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(completions, handle)
    os.replace(tmp_path, path)  # atomic on the same filesystem: never a half-written file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--modality", default="rgb", choices=["depth", "rgb"],
                        help="Same default as build_teacher_cache.py: the teacher "
                             "screens as an RGB model (§9.2/§10.3).")
    parser.add_argument("--representation", default="replicated",
                        choices=["replicated", "gradient"])
    parser.add_argument("--limit", type=int,
                        help="First N rows only — size or sanity-check before "
                             "committing to a full-split pass.")
    parser.add_argument("--out-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500,
                        help="Flush the partial completions table to disk after "
                             "this many new rows, so a restart loses at most this many.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate every row even if already present in a "
                             "partial completions table.")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.teacher)
    teacher_revision = getattr(AutoConfig.from_pretrained(args.teacher), "_commit_hash", None)
    key = build_cache_key(processor, args.split, args.teacher, teacher_revision)
    directory = cache_directory(args.out_root, key)
    os.makedirs(directory, exist_ok=True)
    write_cache_key(directory, key)
    print(f"cache directory: {directory}", flush=True)
    print(json.dumps(key.describe(), indent=2), flush=True)

    rows = load_rows(args.split, args.limit)
    completions = {} if args.overwrite else load_partial(directory)
    resumed = len(completions) if not args.overwrite else 0
    todo = [row for row in rows if row["question_id"] not in completions]
    if resumed:
        print(f"resuming: {resumed} of {len(rows)} already generated; "
              f"{len(todo)} to do", flush=True)

    if not todo:
        print("nothing to generate — cache is already complete for this key", flush=True)
        return

    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"loaded; {torch.cuda.max_memory_allocated() / 1e9:.2f} GB allocated", flush=True)

    started = time.time()
    since_save = 0
    for index, row in enumerate(todo):
        image = build_image(row, args.modality, args.representation)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"{row['question']}\n{PROMPT_SUFFIX}"}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        inputs = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        completion = processor.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions[row["question_id"]] = completion.strip().replace("\n", " ")
        since_save += 1

        if since_save >= args.save_every:
            save_partial(directory, completions)
            since_save = 0

        if (index + 1) % args.log_every == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{len(todo)}  {rate:.2f} it/s  "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    save_partial(directory, completions)
    elapsed = time.time() - started
    summary = {
        "teacher": args.teacher, "split": args.split, "modality": args.modality,
        "rows_generated": len(todo), "rows_resumed": resumed,
        "rows_total": len(completions), "elapsed_minutes": round(elapsed / 60, 2),
        "rows_per_second": round(len(todo) / elapsed, 3) if elapsed else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "cache_key": key.describe(), "cache_digest": key.digest(),
    }
    with open(os.path.join(directory, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
