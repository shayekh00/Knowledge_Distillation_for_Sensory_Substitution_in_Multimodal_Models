"""Renders the committed audit report (§8.3: results under `audit/`, annotator
IDs anonymised).

Usage::

    python -m tools.audit_app.report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.audit_app.audit_items import load_audit_items
from tools.audit_app.audit_store import (
    anonymize_annotator_ids,
    compute_stats,
    load_all_responses,
    render_report_markdown,
)
from tools.audit_app.main import AUDIT_DIR, AUDIT_ITEMS_CSV, RESPONSES_DIR, SCENE_INDEX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-items-csv", type=Path, default=AUDIT_ITEMS_CSV)
    parser.add_argument("--responses-dir", type=Path, default=RESPONSES_DIR)
    parser.add_argument("--out-dir", type=Path, default=AUDIT_DIR / "results")
    args = parser.parse_args()

    items = load_audit_items(args.audit_items_csv, SCENE_INDEX)
    responses_by_annotator = load_all_responses(args.responses_dir)
    if not responses_by_annotator:
        raise SystemExit(f"No responses found under {args.responses_dir}")

    stats = compute_stats(items, responses_by_annotator)
    alias = anonymize_annotator_ids(responses_by_annotator)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.md").write_text(render_report_markdown(stats, alias), encoding="utf-8")
    (args.out_dir / "stats.json").write_text(
        json.dumps([stat.__dict__ for stat in stats], indent=2), encoding="utf-8"
    )

    # Anonymised copies of the raw responses, safe to commit alongside the report.
    for real_id, anon_id in alias.items():
        responses = responses_by_annotator[real_id]
        lines = [
            json.dumps({**response.__dict__, "annotator_id": anon_id})
            for response in responses.values()
        ]
        (args.out_dir / f"responses_{anon_id}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    print(f"Wrote {args.out_dir / 'report.md'}")
    for stat in stats:
        print(f"  {stat.question_type}: gold={stat.gold_accuracy} ambiguous={stat.ambiguous_share} "
              f"kappa={stat.cohen_kappa} meets_acceptance={stat.meets_acceptance}")


if __name__ == "__main__":
    main()
