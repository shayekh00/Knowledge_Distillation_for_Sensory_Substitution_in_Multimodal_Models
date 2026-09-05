"""Persistence and scoring for single-reviewer gold verification (§8.3).

The reviewer's verdicts live in an append-only JSONL file under
``audit/responses/<annotator_id>.jsonl`` — one line per submission. Re-judging
an item just appends another line; readers keep only the last line per
question_id, so the file is both the log and, read back, the current state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from tools.audit_app.audit_items import AuditItem, canonicalize_answer

VERDICTS = ("correct", "incorrect", "ambiguous")

# §8.3 acceptance thresholds.
MIN_GOLD_ACCURACY = 0.95
MAX_AMBIGUOUS_SHARE = 0.03


@dataclass(frozen=True)
class AuditResponse:
    question_id: str
    annotator_id: str
    own_answer: str
    verdict: str
    notes: str
    answered_at_utc: str
    own_answer_raw: str = ""  # exactly what the annotator typed, before spelling correction

    @staticmethod
    def new(question_id: str, annotator_id: str, own_answer: str, verdict: str, notes: str,
            own_answer_raw: str = "") -> "AuditResponse":
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        return AuditResponse(
            question_id=question_id,
            annotator_id=annotator_id,
            own_answer=own_answer,
            verdict=verdict,
            notes=notes,
            answered_at_utc=datetime.now(timezone.utc).isoformat(),
            own_answer_raw=own_answer_raw or own_answer,
        )


def _response_path(responses_dir: Path, annotator_id: str) -> Path:
    return responses_dir / f"{annotator_id}.jsonl"


def append_response(responses_dir: Path, response: AuditResponse) -> None:
    responses_dir.mkdir(parents=True, exist_ok=True)
    path = _response_path(responses_dir, response.annotator_id)
    with path.open("a", encoding="utf-8") as response_file:
        response_file.write(json.dumps(asdict(response)) + "\n")


def load_responses(responses_dir: Path, annotator_id: str) -> dict[str, AuditResponse]:
    """question_id -> that annotator's latest response, or {} if none yet."""
    path = _response_path(responses_dir, annotator_id)
    if not path.is_file():
        return {}
    latest: dict[str, AuditResponse] = {}
    with path.open(encoding="utf-8") as response_file:
        for line in response_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            latest[record["question_id"]] = AuditResponse(**record)
    return latest


def list_annotator_ids(responses_dir: Path) -> list[str]:
    if not responses_dir.is_dir():
        return []
    return sorted(path.stem for path in responses_dir.glob("*.jsonl"))


def load_all_responses(responses_dir: Path) -> dict[str, dict[str, AuditResponse]]:
    return {
        annotator_id: load_responses(responses_dir, annotator_id)
        for annotator_id in list_annotator_ids(responses_dir)
    }


def progress_for(items: list[AuditItem], responses: dict[str, AuditResponse]) -> dict:
    by_type: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = by_type.setdefault(item.question_type, {"total": 0, "answered": 0})
        bucket["total"] += 1
        if item.question_id in responses:
            bucket["answered"] += 1
    total = len(items)
    answered = sum(1 for item in items if item.question_id in responses)
    return {"total": total, "answered": answered, "by_type": by_type}


@dataclass(frozen=True)
class TypeStats:
    question_type: str
    n_sampled: int
    n_verdicts: int
    gold_accuracy: float | None
    human_accuracy_vs_gold: float | None
    ambiguous_share: float | None
    meets_acceptance: bool | None


def compute_stats(items: list[AuditItem], responses_by_annotator: dict[str, dict[str, AuditResponse]]) -> list[TypeStats]:
    """Return one verification row per question type.

    The declared protocol uses exactly one reviewer. Multiple response files
    are rejected so the report cannot silently pool non-independent verdicts
    while describing them as a single-reviewer verification.

    `human_accuracy_vs_gold` (`own_answer` vs. gold) is scoped to responses
    that actually carry an `own_answer`: the UI only asks for one when the
    annotator marks an item "incorrect" (they type what they believe the
    right answer is), so it now reads as "of the corrections offered, how
    many turned out to already match gold" — a consistency check, not the
    independent-agreement metric this field measured before the reveal-gold
    workflow changed (see docs/DATASET_CREATION_PLAN.md §8.3 discussion).
    """
    reviewer_ids = sorted(responses_by_annotator)
    if len(reviewer_ids) > 1:
        raise ValueError(
            "Single-reviewer gold verification requires exactly one reviewer; "
            f"found {len(reviewer_ids)}"
        )
    reviewer_responses = responses_by_annotator.get(reviewer_ids[0], {}) if reviewer_ids else {}
    items_by_type: dict[str, list[AuditItem]] = {}
    for item in items:
        items_by_type.setdefault(item.question_type, []).append(item)

    results: list[TypeStats] = []
    for question_type, type_items in items_by_type.items():
        verdicts: list[str] = []
        correct_answers = 0
        answer_judgements = 0

        for item in type_items:
            response = reviewer_responses.get(item.question_id)
            if response is None:
                continue
            verdicts.append(response.verdict)
            if response.own_answer.strip():
                answer_judgements += 1
                if canonicalize_answer(response.own_answer) == canonicalize_answer(item.answer):
                    correct_answers += 1

        n_verdicts = len(verdicts)
        gold_accuracy = verdicts.count("correct") / n_verdicts if n_verdicts else None
        ambiguous_share = verdicts.count("ambiguous") / n_verdicts if n_verdicts else None
        human_accuracy = correct_answers / answer_judgements if answer_judgements else None

        meets_acceptance = None
        if n_verdicts == len(type_items):
            meets_acceptance = (
                gold_accuracy is not None and ambiguous_share is not None
                and gold_accuracy >= MIN_GOLD_ACCURACY
                and ambiguous_share <= MAX_AMBIGUOUS_SHARE
            )

        results.append(TypeStats(
            question_type=question_type,
            n_sampled=len(type_items),
            n_verdicts=n_verdicts,
            gold_accuracy=gold_accuracy,
            human_accuracy_vs_gold=human_accuracy,
            ambiguous_share=ambiguous_share,
            meets_acceptance=meets_acceptance,
        ))
    return sorted(results, key=lambda row: row.question_type)


def anonymize_annotator_ids(responses_by_annotator: dict[str, dict[str, AuditResponse]]) -> dict[str, str]:
    """Map the reviewer id to an anonymised label for the committed report."""
    def earliest_timestamp(annotator_id: str) -> str:
        responses = responses_by_annotator[annotator_id]
        return min((r.answered_at_utc for r in responses.values()), default="")

    ordered = sorted(responses_by_annotator, key=earliest_timestamp)
    return {annotator_id: f"R{i + 1}" for i, annotator_id in enumerate(ordered)}


def _format_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def render_report_markdown(stats: list[TypeStats], annotator_alias: dict[str, str]) -> str:
    lines = [
        "# VQA-SUNRGBD-v2 — single-reviewer gold verification report",
        "",
        f"Reviewer: {', '.join(sorted(annotator_alias.values())) or '(none yet)'}",
        "Protocol: one reviewer inspects the RGB image, evidence overlay, question, and gold answer, then records correct, incorrect, or ambiguous.",
        "This verifies sampled gold labels; it does not measure inter-rater reliability.",
        f"Acceptance rule (§8.3): gold accuracy ≥ {MIN_GOLD_ACCURACY:.0%} and ambiguous share ≤ {MAX_AMBIGUOUS_SHARE:.0%}.",
        "",
        "| Type | Sampled | Reviewed | Gold accuracy | Corrections matching gold | Ambiguous | Meets acceptance |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in stats:
        acceptance_cell = "yes" if row.meets_acceptance else ("no" if row.meets_acceptance is False else "—")
        lines.append(
            f"| {row.question_type} | {row.n_sampled} | {row.n_verdicts} | "
            f"{_format_pct(row.gold_accuracy)} | {_format_pct(row.human_accuracy_vs_gold)} | "
            f"{_format_pct(row.ambiguous_share)} | {acceptance_cell} |"
        )
    lines.append("")
    return "\n".join(lines)
