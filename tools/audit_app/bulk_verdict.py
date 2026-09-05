"""Records one verdict across many audit items at once.

Intended for a *type-level* judgment: after reviewing enough of a question
type to conclude that model/gold disagreements there are not diagnostic
(e.g. the depth-derived types, where gold comes from measured depth while
the model guesses 3-D structure from a single RGB frame), this writes that
conclusion across the matching items instead of clicking through each one.

Caveat worth keeping in mind when reporting audit numbers: rows written
here reflect a judgment about the *type*, not an inspection of each
individual item, even though they are stored in the same format as
hand-reviewed rows. Items already judged in the UI are never overwritten.
Everything is appended to the same append-only log, so a bulk run is
identifiable by its shared timestamp and can be undone (see --undo).

Usage::

    python -m tools.audit_app.bulk_verdict \
        --types identify_superlative,relative_depth,nearest_object \
        --only-disagreements --verdict correct --dry-run
    # then drop --dry-run to write

    python -m tools.audit_app.bulk_verdict --undo <iso-timestamp-prefix>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from tools.audit_app.audit_store import VERDICTS, AuditResponse, append_response, load_responses

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path(os.environ.get("AUDIT_DIR", PROJECT_ROOT / "audit"))
AUDIT_ITEMS_CSV = AUDIT_DIR / "audit_items.csv"
MODEL_ANSWERS_CSV = AUDIT_DIR / "model_answers.csv"
RESPONSES_DIR = AUDIT_DIR / "responses"


def select_items(types: list, variants: list, only_disagreements: bool) -> pd.DataFrame:
    items = pd.read_csv(AUDIT_ITEMS_CSV)
    if types:
        items = items[items["question_type"].isin(types)]
    if variants:
        if "variant" not in items.columns:
            raise SystemExit(f"{AUDIT_ITEMS_CSV} is missing required column 'variant'.")
        items = items[items["variant"].fillna("").isin(variants)]
    if only_disagreements:
        if not MODEL_ANSWERS_CSV.is_file():
            raise SystemExit(f"{MODEL_ANSWERS_CSV} not found — run tools.audit_app.model_pass first.")
        model = (pd.read_csv(MODEL_ANSWERS_CSV)
                 .drop_duplicates(subset="question_id", keep="last"))
        disagreeing = set(model.loc[~model["agrees_with_gold"], "question_id"])
        items = items[items["question_id"].isin(disagreeing)]
    return items


def undo(annotator_id: str, timestamp_prefix: str) -> None:
    path = RESPONSES_DIR / f"{annotator_id}.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in lines
            if not json.loads(line)["answered_at_utc"].startswith(timestamp_prefix)]
    removed = len(lines) - len(kept)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"Removed {removed} row(s) whose timestamp starts with {timestamp_prefix!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", default="solo")
    parser.add_argument("--types", default="", help="Comma-separated question_types.")
    parser.add_argument("--variants", default="", help="Comma-separated variants within the selected types.")
    parser.add_argument("--verdict", default="correct", choices=VERDICTS)
    parser.add_argument("--only-disagreements", action="store_true",
                         help="Restrict to items where the model pass disagreed with gold.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--undo", metavar="TIMESTAMP_PREFIX",
                         help="Delete previously written rows by answered_at_utc prefix.")
    args = parser.parse_args()

    if args.undo:
        undo(args.annotator, args.undo)
        return

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    selected = select_items(types, variants, args.only_disagreements)
    already_judged = load_responses(RESPONSES_DIR, args.annotator)
    pending = selected[~selected["question_id"].isin(already_judged)]
    skipped = len(selected) - len(pending)

    print(f"Selected {len(selected)} item(s); {skipped} already judged in the UI (left untouched); "
          f"{len(pending)} to write as verdict={args.verdict!r} for annotator {args.annotator!r}.")
    print(pending.groupby("question_type").size().to_string() if len(pending) else "(nothing to write)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if pending.empty:
        return

    written = 0
    for question_id in pending["question_id"]:
        append_response(RESPONSES_DIR, AuditResponse.new(
            question_id=question_id, annotator_id=args.annotator,
            own_answer="", verdict=args.verdict, notes=args.notes,
        ))
        written += 1

    responses = load_responses(RESPONSES_DIR, args.annotator)
    stamps = sorted(r.answered_at_utc for r in responses.values())
    print(f"\nWrote {written} row(s). Undo with:\n"
          f"  python -m tools.audit_app.bulk_verdict --undo {stamps[-1][:19]}")


if __name__ == "__main__":
    main()
