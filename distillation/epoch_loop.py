"""Shared per-epoch validation + early stopping (protocol §7.3).

§7.3: "Stage-two stop: up to 5 epochs; stop when validation macro accuracy
fails to improve over 2 consecutive evaluation points. Every compared row gets
equal checkpoint-evaluation opportunities." (Author raised the cap to 10 for
this run — experiment_protocol.md §13, 2026-09-06 — patience unchanged at 2.)

Neither `train_student.py` nor `train_kd.py` implemented this before: both
trained a fixed 1 epoch, which was matched between CE and KD but never
compared against what either would do with more training. This module is what
makes "equal checkpoint-evaluation opportunities" actually true going forward
— both scripts call the exact same validation path, at the same per-epoch
cadence, so neither is stopped on a differently-defined signal.

Validation is the real task metric, not the training loss. §7.3 says
"validation macro accuracy" specifically, and §4's fairness rule forbids
comparing CE and KD "by comparing differently scaled CE and KD losses" — a raw
loss number is exactly the thing that rule exists to keep out of a stopping
decision. So this generates real greedy completions and scores them through
the same `evaluate.score_predictions` every reported number in this project
goes through, using the model already resident in memory rather than
reloading a checkpoint from disk each epoch the way `zero_shot_inference.py`
does for a *finished* run.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate import (  # noqa: E402
    RELEASE_DIR,
    VOCAB_DIR,
    load_release_split,
    macro_accuracy,
    score_predictions,
)
from evaluation.zero_shot_inference import PROMPT_STYLES  # noqa: E402
from vocab import load_canonical_vocab, load_synonyms  # noqa: E402

MAX_NEW_TOKENS = 16  # matches evaluation/zero_shot_inference.py exactly


def generate_val_predictions(model, processor, rows, images, prompt_style="terse"):
    """Greedy completions for `rows`, using the model already resident in
    memory mid-training.

    Decoding matches `zero_shot_inference.py` exactly (`do_sample=False`,
    `max_new_tokens=16`, `enable_thinking=False`, the `terse` suffix — the
    same prompt frozen in protocol §8.3) so a per-epoch number sits on the
    same footing as the val CSVs every other score in this project comes from.
    Restores whatever `model.training` mode it found, since the caller still
    has more epochs of training to do.
    """
    import torch

    suffix = PROMPT_STYLES[prompt_style]
    was_training = model.training
    model.eval()
    compute_dtype = next(model.parameters()).dtype
    predictions = []
    try:
        with torch.inference_mode():
            for row, image in zip(rows, images):
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": (f"{row['question']}\n{suffix}" if suffix
                                              else row["question"])}]}]
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False)
                inputs = processor(images=image, text=prompt,
                                   return_tensors="pt").to(model.device)
                # Same float/int split as zero_shot_inference.py: pixel_values
                # comes back float32 from the processor regardless of model
                # dtype, and only the floating inputs should be recast.
                inputs = {key: (value.to(compute_dtype) if value.is_floating_point()
                                else value) for key, value in inputs.items()}
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False)
                completion = processor.decode(
                    generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                predictions.append((row["question_id"], completion.strip().replace("\n", " ")))
    finally:
        if was_training:
            model.train()
    return predictions


def score_val_macro(predictions, split: str = "val") -> float:
    """Score `[(question_id, prediction), ...]` exactly as `evaluate.py`
    scores any prediction file, and return the macro accuracy.

    **Missing predictions count as wrong**, because that is what `evaluate.py`
    does — it scores against every gold row in the split, not against the rows
    you happened to supply. So a partial prediction set does not yield "the
    accuracy on that subset", it yields the subset's accuracy diluted by the
    whole split, and the result looks like catastrophic failure rather than
    partial coverage.

    That has now produced a wrong reading three times in this project: a
    `--limit 300` inference run scored 7.1% against a 30.3% chance floor, and a
    `--val-limit 8` stage-F run scored 0.06% and was briefly taken as evidence
    that the stage had no usable metric at all (it scores 34.3% at full
    coverage). Each time the number was arithmetically correct and completely
    misleading. So coverage is checked and announced here rather than left for
    someone to notice: partial coverage is legitimate for a smoke test, but it
    must never be read as, or compared against, a real result.
    """
    gold = load_release_split(split, RELEASE_DIR)
    synonym_map = load_synonyms(os.path.join(VOCAB_DIR, "synonyms.csv"))
    canonical_vocab = load_canonical_vocab(os.path.join(VOCAB_DIR, "canonical_objects.csv"))
    frame = pd.DataFrame(predictions, columns=["question_id", "prediction"])
    if len(frame) < len(gold):
        print(f"  !! PARTIAL COVERAGE: {len(frame)} predictions scored against "
              f"{len(gold)} gold rows in '{split}'. The {len(gold) - len(frame)} "
              f"missing rows count as wrong, so the macro below is a smoke-test "
              f"number only — it is NOT comparable to any recorded result and must "
              f"not be quoted.", flush=True)
    scores = score_predictions(gold, frame, synonym_map, canonical_vocab)
    return macro_accuracy(scores)


class EarlyStopper:
    """§7.3's rule: stop when validation macro fails to improve over
    `patience` consecutive evaluation points, up to `max_epochs`.

    Tracks the best epoch seen, because "early stopping" means keeping the
    best checkpoint once training stops, not whichever epoch happened to
    trigger the patience limit — those are usually not the same epoch.
    """

    def __init__(self, max_epochs: int, patience: int):
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.max_epochs = max_epochs
        self.patience = patience
        self.best_epoch = None
        self.best_macro = float("-inf")
        self.epochs_since_improvement = 0
        self.history: list[dict] = []

    def step(self, epoch: int, val_macro: float) -> bool:
        """Record one epoch's validation result.

        Returns True if training should stop *after* this epoch — either
        `patience` consecutive epochs without improvement, or `max_epochs`
        reached.
        """
        self.history.append({"epoch": epoch, "val_macro": val_macro})
        if val_macro > self.best_macro:
            self.best_macro = val_macro
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        stop_on_patience = self.epochs_since_improvement >= self.patience
        stop_on_cap = (epoch + 1) >= self.max_epochs
        return stop_on_patience or stop_on_cap
