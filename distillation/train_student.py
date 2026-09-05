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
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.depth_input import decode_metric_depth, depth_to_student_input  # noqa: E402
from distillation.losses import IGNORE_INDEX, masked_cross_entropy  # noqa: E402

PROMPT_SUFFIX = "Answer in one or two words. No explanation."


def load_rows(split: str, limit: int | None = None) -> list:
    path = os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2", "rule_based",
                        f"{split}.csv")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def build_image(row, modality: str, representation: str):
    from PIL import Image
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    if modality == "rgb":
        return Image.open(os.path.join(dataset_dir, row["image_path"])).convert("RGB")
    metres = decode_metric_depth(os.path.join(dataset_dir, row["depth_path"]))
    return Image.fromarray(depth_to_student_input(metres, representation))


def build_example(processor, row, image, device):
    """One training example, with labels masked to the answer span.

    The prompt is tokenized on its own first so its exact token length is known;
    everything before the answer is then set to IGNORE_INDEX. Masking by string
    search, or masking only padding, is how the legacy path ended up training on
    the question.
    """
    import torch

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{row['question']}\n{PROMPT_SUFFIX}"}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                tokenize=False)
    answer_text = str(row["answer"]) + processor.tokenizer.eos_token

    prompt_inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    full_inputs = processor(images=image, text=prompt_text + answer_text,
                            return_tensors="pt")

    prompt_length = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_length] = IGNORE_INDEX
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX

    if (labels != IGNORE_INDEX).sum() == 0:
        return None                      # nothing supervised; skip rather than crash
    batch = {key: value.to(device) for key, value in full_inputs.items()}
    batch["labels"] = labels.to(device)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--recipe", default="B3")
    parser.add_argument("--modality", choices=["depth", "rgb"], default="depth")
    parser.add_argument("--representation", choices=["replicated", "gradient"],
                        default="replicated")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Train on the first N rows only.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--out", default="runs/pilot/B3_depth")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows("train", args.limit)
    print(f"{len(rows)} training rows; recipe={args.recipe}; modality={args.modality}",
          flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")

    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    order = list(range(len(rows)))
    random.shuffle(order)
    started = time.time()
    step = seen = skipped = 0
    running = 0.0
    history = []

    for epoch in range(args.epochs):
        for position, index in enumerate(order):
            row = rows[index]
            try:
                image = build_image(row, args.modality, args.representation)
                batch = build_example(processor, row, image, model.device)
            except Exception as error:                       # unreadable frame, bad row
                skipped += 1
                if skipped <= 3:
                    print(f"  skipped {row['question_id']}: {error}", flush=True)
                continue
            if batch is None:
                skipped += 1
                continue

            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = masked_cross_entropy(outputs.logits, labels)
            (loss / args.accumulation).backward()
            running += float(loss.detach())
            seen += 1

            if seen % args.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            if seen % args.log_every == 0:
                elapsed = time.time() - started
                rate = seen / elapsed
                mean_loss = running / args.log_every
                running = 0.0
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  epoch {epoch} {position + 1}/{len(order)}  step {step}  "
                      f"loss {mean_loss:.4f}  {rate:.2f} ex/s  peak {peak:.2f} GB",
                      flush=True)
                history.append({"step": step, "examples": seen, "loss": mean_loss,
                                "examples_per_second": rate, "peak_vram_gb": peak})

    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1e9
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(os.path.join(args.out, "adapter"))

    resources = {
        "recipe": args.recipe, "model": args.model, "modality": args.modality,
        "seed": args.seed, "epochs": args.epochs, "examples_seen": seen,
        "examples_skipped": skipped, "optimizer_steps": step,
        "learning_rate": args.learning_rate, "accumulation": args.accumulation,
        "lora_rank": args.lora_rank,
        "elapsed_minutes": round(elapsed / 60, 2),
        "examples_per_second": round(seen / elapsed, 3) if elapsed else None,
        "peak_vram_gb": round(peak, 2),
        "gpu": torch.cuda.get_device_name(0),
        "pilot": True,
        "projected_minutes_full_train_epoch": (
            round((15278 / (seen / elapsed)) / 60, 1) if seen and elapsed else None),
    }
    with open(os.path.join(args.out, "resource_usage.json"), "w", encoding="utf-8") as handle:
        json.dump({"resources": resources, "history": history}, handle, indent=2)

    print(json.dumps(resources, indent=2), flush=True)
    print(f"adapter saved to {args.out}/adapter", flush=True)


if __name__ == "__main__":
    main()
