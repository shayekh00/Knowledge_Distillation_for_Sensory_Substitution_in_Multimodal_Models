"""Zero-shot reference rows B1/B2 and the teacher-suitability check (§9.2, §10.3).

These rows need **no training**: they run a pretrained model over the frozen
release and write a prediction CSV. That makes them the only part of the
experiment matrix that is fully available before the 24 GB card returns, and it is
worth doing early for three reasons beyond the numbers themselves:

* it exercises the repaired evaluator, the corrected depth decoder, and the frozen
  prompt end-to-end against a real model rather than a fixture;
* it answers §10.3's teacher-suitability question, which is a *precondition* of
  the study. "A large RGB teacher may be weaker than a depth student on measured
  depth relations" — if that is true here, the premise needs revisiting before any
  training pipeline is built;
* it settles the §4.1 provenance question about the historical teacher figure,
  whose input modality the old table never recorded.

Runs on the **val** split by default. Test is evaluated once, after settings are
locked (`experiment_protocol.md` §5), and a zero-shot diagnostic is not that
moment.

Depth inputs are built through `distillation.depth_input`, so this path uses the
same decoding the gold answers came from — not the legacy per-image min-max that
disagrees with it on 20% of xtion frames.

Usage::

    python evaluation/zero_shot_inference.py --model Qwen/Qwen3.5-0.8B \
        --modality depth --split val --out runs/pilot/B1_depth_val.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.depth_input import decode_metric_depth, depth_to_student_input  # noqa: E402

# The frozen instruction from plan §6.4. It replaces the legacy "single word or
# number", which conflicts with legitimate multiword answers such as "tissue box".
PROMPT_SUFFIX = ("Answer with only the short answer: yes, no, left, right, or the "
                 "object name, as appropriate. Use no explanation.")

# The §6.4 wording enumerates the legal answers, and a 0.8B student copies the
# enumeration back verbatim ("yes, no, left, right, or the object name") instead
# of answering. Alternatives are kept here so the choice is made on development
# data and then frozen, rather than retrofitted after seeing test results.
PROMPT_STYLES = {
    "enumerated": PROMPT_SUFFIX,
    "terse": "Answer in one or two words. No explanation.",
    "none": "",
}

MAX_NEW_TOKENS = 16


def load_rows(release_csv: str, limit: int | None = None) -> list:
    with open(release_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def build_image(row, dataset_dir: str, modality: str, representation: str):
    """The visual input for one row. Depth is decoded to metres first."""
    from PIL import Image
    if modality == "rgb":
        return Image.open(os.path.join(dataset_dir, row["image_path"])).convert("RGB")
    depth_metres = decode_metric_depth(os.path.join(dataset_dir, row["depth_path"]))
    return Image.fromarray(depth_to_student_input(depth_metres, representation))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model id.")
    parser.add_argument("--modality", choices=["depth", "rgb"], default="depth")
    parser.add_argument("--representation", choices=["replicated", "gradient"],
                        default="replicated")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", required=True, help="Prediction CSV to write.")
    parser.add_argument("--limit", type=int, help="Only the first N rows (smoke test).")
    parser.add_argument("--prompt-style", choices=sorted(PROMPT_STYLES), default="enumerated",
                        help="Instruction wording; frozen before the main runs (§6.4).")
    parser.add_argument("--thinking", choices=["off", "on"], default="off",
                        help="Qwen3.5 is a reasoning model whose chat template emits "
                             "chain-of-thought by default. Left on, a 16-token budget "
                             "captures truncated reasoning instead of an answer.")
    parser.add_argument("--adapter", help="LoRA adapter directory to load on top of "
                        "the base model. Without it this is a zero-shot run.")
    parser.add_argument("--quantize", choices=["none", "nf4", "int8"], default="none")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    release_csv = os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2",
                               "rule_based", f"{args.split}.csv")
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    rows = load_rows(release_csv, args.limit)
    suffix = PROMPT_STYLES[args.prompt_style]
    print(f"{len(rows)} rows from {args.split}; model={args.model}; "
          f"modality={args.modality}", flush=True)

    load_kwargs = {"torch_dtype": getattr(torch, args.dtype), "device_map": "cuda:0"}
    if args.quantize != "none":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
            if args.quantize == "nf4" else BitsAndBytesConfig(load_in_8bit=True))
        load_kwargs.pop("torch_dtype")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"loaded adapter {args.adapter}", flush=True)
    model.eval()
    peak_before = torch.cuda.max_memory_allocated() / 1e9
    print(f"loaded; {peak_before:.2f} GB allocated", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "prediction"])
        for index, row in enumerate(rows):
            image = build_image(row, dataset_dir, args.modality, args.representation)
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": (f"{row['question']}\n{suffix}" if suffix
                                          else row["question"])}]}]
            template_kwargs = {} if args.thinking == "on" else {"enable_thinking": False}
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=False, **template_kwargs)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False)
            completion = processor.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            writer.writerow([row["question_id"], completion.strip().replace("\n", " ")])
            if (index + 1) % 100 == 0:
                rate = (index + 1) / (time.time() - started)
                print(f"  {index + 1}/{len(rows)}  {rate:.1f} it/s  "
                      f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    elapsed = time.time() - started
    print(f"wrote {args.out}", flush=True)
    print(f"elapsed {elapsed / 60:.1f} min; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
