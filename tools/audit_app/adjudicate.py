"""Second-opinion pass: re-asks the model about items it disagreed with,
this time showing it both its own answer and the gold, and asking it to
judge between them.

Why this needs a control arm to mean anything
---------------------------------------------
Telling a model "my data says X" reliably pushes it toward X, whatever X
is — so "the model agreed with our gold on re-ask" is not evidence our
gold is right unless we know the model would *not* have agreed with an
answer we made up. Every item is therefore adjudicated twice, with
prompts identical except for the asserted answer:

  gold arm   — asserted answer is the dataset's gold label
  decoy arm  — asserted answer is a plausible wrong one

For identify_superlative the decoy comes free and is ideal: the
generator's own evidence records `runner_up_concept`, the object that
*lost* the margin comparison (second largest / second closest). It is
from the same scene, of the same kind, and wrong by construction.

Reading the result:
  * gold acceptance >> decoy acceptance → the model is discriminating,
    and its swing toward gold is real evidence about the labels.
  * gold acceptance ≈ decoy acceptance → the model just defers to
    whatever it is told; the adjudication says nothing about the labels
    and its agreement number must not be reported as validation.

The model is never ground truth here either way (same caveat as
model_pass.py). This measures whether a *second* look, with both
candidates on the table, changes the picture — nothing more.

Usage::

    python -m tools.audit_app.adjudicate                     # all disagreements
    python -m tools.audit_app.adjudicate --limit 10          # smoke test
    python -m tools.audit_app.adjudicate --type left_right   # a different type

Resumable: results are appended per (question_id, arm) and a rerun skips
pairs already recorded with a usable answer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from tools.audit_app.agreement import answers_agree, canonical_answer_form
from tools.audit_app.model_pass import (
    DEFAULT_MODEL,
    _load_vocab_tables,
    ask_model_with_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIT_DIR = Path(os.environ.get("AUDIT_DIR", PROJECT_ROOT / "audit"))
AUDIT_ITEMS_CSV = AUDIT_DIR / "audit_items.csv"
MODEL_ANSWERS_CSV = AUDIT_DIR / "model_answers.csv"
ADJUDICATION_CSV = AUDIT_DIR / "adjudication.csv"

OUTPUT_COLUMNS = [
    "question_id", "question_type", "arm", "asserted_answer", "original_answer",
    "final_answer", "kept_original", "accepted_asserted", "reasoning", "error",
]

# Deliberately mirrors the wording a reviewer would use, including the
# explicit instruction not to defer — the decoy arm exists precisely to
# measure how much that instruction is worth.
ADJUDICATION_PROMPT = (
    "Look at this image and consider the question: {question}\n\n"
    "You previously answered: \"{original}\".\n"
    "My data says the answer is: \"{asserted}\".\n\n"
    "Think properly and make an unbiased judgement. Do not take my data for granted — "
    "reason from what you can actually see in the image. One of these answers may be "
    "wrong, or both may be. Decide which answer is correct on the evidence of the image "
    "itself.\n\n"
    "Reply with only your final answer, as a single word or short noun phrase."
)

_write_lock = threading.Lock()


def _already_done(path: Path) -> set:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {
            (row["question_id"], row["arm"]) for row in csv.DictReader(csv_file)
            if (row.get("final_answer") or "").strip() and not (row.get("error") or "").strip()
        }


def _append_row(path: Path, row: dict) -> None:
    with _write_lock:
        is_new = not path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def decoy_for(row: dict, synonym_map: dict, canonical_vocab: dict) -> str | None:
    """A plausible-but-wrong answer to assert in the control arm.

    `runner_up_concept` is the generator's own second-place object — it
    lost the margin comparison, so it is wrong by construction while
    remaining a same-scene, same-kind candidate. Unusable if it happens to
    coincide with either answer already on the table, since the arms must
    differ only in what is asserted.
    """
    try:
        evidence = json.loads(row["evidence"])
    except (json.JSONDecodeError, TypeError):
        return None
    runner_up = evidence.get("runner_up_concept")
    if not runner_up:
        return None
    decoy = str(runner_up).replace("_", " ")
    qtype = row["question_type"]
    for taken in (row["answer"], row["model_answer"]):
        if answers_agree(decoy, str(taken), qtype, synonym_map, canonical_vocab):
            return None
    return decoy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--type", default="identify_superlative",
                        help="Question type to adjudicate (default: identify_superlative).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pending calls.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(str(PROJECT_ROOT / ".env"))
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing from .env")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    synonym_map, canonical_vocab = _load_vocab_tables()

    items = pd.read_csv(AUDIT_ITEMS_CSV, dtype={"answer": str})
    model_answers = pd.read_csv(MODEL_ANSWERS_CSV).drop_duplicates("question_id", keep="last")
    merged = items.merge(
        model_answers[["question_id", "model_answer", "agrees_with_gold", "error"]],
        on="question_id",
    )
    merged = merged[(merged["question_type"] == args.type) & merged["error"].isna()]
    disagreements = merged[~merged["agrees_with_gold"].astype(bool)]

    pending = []
    no_decoy = 0
    for row in disagreements.to_dict(orient="records"):
        pending.append((row, "gold", str(row["answer"])))
        decoy = decoy_for(row, synonym_map, canonical_vocab)
        if decoy is None:
            no_decoy += 1
        else:
            pending.append((row, "decoy", decoy))

    done = _already_done(ADJUDICATION_CSV)
    pending = [p for p in pending if (p[0]["question_id"], p[1]) not in done]
    if args.limit:
        pending = pending[:args.limit]

    print(f"{len(disagreements)} disagreement(s) of type {args.type}; "
          f"{no_decoy} without a usable decoy; {len(pending)} call(s) to make")
    if not pending:
        report()
        return

    progress = {"n": 0}

    def process(job) -> None:
        row, arm, asserted = job
        original = str(row["model_answer"])
        prompt = ADJUDICATION_PROMPT.format(
            question=row["question"], original=original, asserted=asserted
        )
        final, reasoning, error = ask_model_with_prompt(
            client, args.model, prompt, DATASET_DIR / row["image_path"]
        )
        qtype = row["question_type"]
        kept = bool(final) and answers_agree(final, original, qtype, synonym_map, canonical_vocab)
        accepted = bool(final) and answers_agree(final, asserted, qtype, synonym_map, canonical_vocab)
        _append_row(ADJUDICATION_CSV, {
            "question_id": row["question_id"], "question_type": qtype, "arm": arm,
            "asserted_answer": asserted, "original_answer": original,
            "final_answer": final, "kept_original": kept, "accepted_asserted": accepted,
            "reasoning": reasoning, "error": error,
        })
        with _write_lock:
            progress["n"] += 1
            if progress["n"] % 25 == 0 or progress["n"] == len(pending):
                print(f"  {progress['n']}/{len(pending)} calls done", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, pending))

    report()


def report() -> None:
    if not ADJUDICATION_CSV.is_file():
        return
    frame = pd.read_csv(ADJUDICATION_CSV).drop_duplicates(["question_id", "arm"], keep="last")
    frame = frame[frame["error"].isna()]
    print("\nSecond-opinion outcome, by arm "
          "(the asserted answer is the gold label in one arm, a known-wrong "
          "runner-up in the other; identical prompts otherwise):\n")
    print(f"{'arm':<8} {'n':>5} {'accepted asserted':>19} {'kept its own':>14} {'neither':>9}")
    for arm, group in frame.groupby("arm"):
        accepted = group["accepted_asserted"].mean()
        kept = group["kept_original"].mean()
        neither = 1.0 - accepted - kept
        print(f"{arm:<8} {len(group):>5} {accepted:>18.1%} {kept:>13.1%} {neither:>8.1%}")

    gold = frame[frame["arm"] == "gold"]
    decoy = frame[frame["arm"] == "decoy"]
    if len(gold) and len(decoy):
        gap = gold["accepted_asserted"].mean() - decoy["accepted_asserted"].mean()
        print(f"\nGold-minus-decoy acceptance gap: {gap:+.1%}")
        print("A large positive gap means the model is discriminating, so its swing toward")
        print("gold is informative. A gap near zero means it defers to whatever it is told,")
        print("and the gold-arm number is NOT evidence the labels are right.")


if __name__ == "__main__":
    main()
