"""Runs a vision model over the sampled audit items, to order the human review.

Purpose (and its limits): the model's answer is a *triage signal*, not
ground truth and not a second annotator. It exists so the reviewer meets
the likely-broken items first. The audit sample itself is untouched — every
sampled item still gets reviewed, so the reported gold-error rate stays an
unbiased estimate over a stratified random sample (§8.3). Agreement is
never treated as proof a label is correct: a model and the gold heuristic
can be wrong the same way (both pulled by SUNRGBD's frequency prior), which
is exactly why "review only the disagreements" was rejected as a design.

The question is sent with the benchmark's own released prompt suffix
(Rule Q5: "Answer with a single word or number."), so the numbers here are
also a rough sanity signal about the benchmark itself.

Usage::

    python -m tools.audit_app.model_pass                  # all sampled items
    python -m tools.audit_app.model_pass --limit 20       # smoke test
    python -m tools.audit_app.model_pass --workers 8

Resumable: results are appended per item, and a rerun skips question_ids
already present in the output CSV.
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from tools.audit_app.agreement import answers_agree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIT_DIR = Path(os.environ.get("AUDIT_DIR", PROJECT_ROOT / "audit"))
AUDIT_ITEMS_CSV = AUDIT_DIR / "audit_items.csv"
MODEL_ANSWERS_CSV = AUDIT_DIR / "model_answers.csv"

DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"
ANSWER_FORMAT_INSTRUCTION = "Answer with a single word or number."
# The model reasons before answering; too small a budget returns an empty
# string (all budget spent on reasoning tokens).
MAX_TOKENS = 512
OUTPUT_COLUMNS = [
    "question_id", "question_type", "model", "model_answer",
    "agrees_with_gold", "model_reasoning", "error",
]

_write_lock = threading.Lock()


def _load_vocab_tables():
    import sys
    v2_dir = str(PROJECT_ROOT / "dataset" / "dataset_creation" / "v2")
    if v2_dir not in sys.path:
        sys.path.insert(0, v2_dir)
    from vocab import load_canonical_vocab, load_synonyms

    return (
        load_synonyms(str(DATA_DIR / "vocab" / "synonyms.csv")),
        load_canonical_vocab(str(DATA_DIR / "vocab" / "canonical_objects.csv")),
    )


def _already_done(path: Path) -> set:
    """question_ids that produced a usable answer.

    Rows that errored or came back blank are deliberately *not* counted as
    done, so a rerun retries exactly those. The file is append-only, so a
    retried item gains a second row; readers keep the last row per
    question_id (see _latest_rows / main.py's _load_model_hints).
    """
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {
            row["question_id"] for row in csv.DictReader(csv_file)
            if (row.get("model_answer") or "").strip() and not (row.get("error") or "").strip()
        }


def _latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Last row per question_id — the current state of the append-only log."""
    return frame.drop_duplicates(subset="question_id", keep="last")


def _append_row(path: Path, row: dict) -> None:
    with _write_lock:
        is_new = not path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def ask_model(client: OpenAI, model: str, question: str, image_absolute_path: Path,
               retries: int = 3) -> tuple:
    """Returns (answer_text, reasoning_text, error_text).

    A blank `content` with no exception is a real failure mode here, not an
    answer: the model spends tokens reasoning first, so a tight budget can
    leave nothing for the answer itself. Those retry with a bigger budget
    and, if still blank, come back as error="empty_answer" — so the review
    queue can tell "the model disagreed" (worth a human look) apart from
    "the model produced nothing" (worth nothing).
    """
    return ask_model_with_prompt(
        client, model, f"{question} {ANSWER_FORMAT_INSTRUCTION}", image_absolute_path, retries
    )


def ask_model_with_prompt(client: OpenAI, model: str, prompt: str,
                           image_absolute_path: Path, retries: int = 3) -> tuple:
    """Same contract as ask_model, for callers that build their own prompt
    (see adjudicate.py's second-opinion pass)."""
    encoded_image = base64.b64encode(image_absolute_path.read_bytes()).decode()
    last_error = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                ]}],
                max_tokens=MAX_TOKENS * (attempt + 1),
                temperature=0,
            )
            message = response.choices[0].message
            answer = (message.content or "").strip()
            reasoning = (getattr(message, "reasoning_content", "") or "").strip()
            if answer:
                return answer, reasoning, ""
            last_error = "empty_answer"
        except Exception as error:  # network/rate-limit/server errors
            last_error = str(error)[:300]
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return "", "", last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pending items.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(str(PROJECT_ROOT / ".env"))
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing from .env")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    synonym_map, canonical_vocab = _load_vocab_tables()
    items = pd.read_csv(AUDIT_ITEMS_CSV)
    done = _already_done(MODEL_ANSWERS_CSV)
    pending = items[~items["question_id"].isin(done)]
    if args.limit:
        pending = pending.head(args.limit)

    print(f"{len(done)} already done, {len(pending)} to process with {args.model}")
    if pending.empty:
        return

    completed = {"n": 0, "agree": 0, "failed": 0}

    def process(row: dict) -> None:
        image_path = DATASET_DIR / row["image_path"]
        answer, reasoning, error = ask_model(client, args.model, row["question"], image_path)
        agrees = bool(answer) and answers_agree(
            answer, row["answer"], row["question_type"], synonym_map, canonical_vocab
        )
        _append_row(MODEL_ANSWERS_CSV, {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "model": args.model,
            "model_answer": answer,
            "agrees_with_gold": agrees,
            "model_reasoning": reasoning,
            "error": error,
        })
        with _write_lock:
            completed["n"] += 1
            completed["agree"] += int(agrees)
            completed["failed"] += int(bool(error))
            if completed["n"] % 25 == 0 or completed["n"] == len(pending):
                print(f"  {completed['n']}/{len(pending)} done "
                      f"(agree {completed['agree']}, failed {completed['failed']})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, pending.to_dict(orient="records")))

    results = _latest_rows(pd.read_csv(MODEL_ANSWERS_CSV))
    print(f"\n{len(results)} items scored -> {MODEL_ANSWERS_CSV}")
    print("\nAgreement with gold, by question type "
          "(low agreement = more items to review first, NOT proof of label error):")
    summary = results.groupby("question_type")["agrees_with_gold"].agg(["mean", "count"])
    print(summary.to_string(float_format=lambda value: f"{value:.1%}"))
    failures = int((results["model_answer"].isna() | results["error"].notna()).sum())
    if failures:
        print(f"\n{failures} item(s) still have no answer; rerun the command to retry just those.")


if __name__ == "__main__":
    main()
