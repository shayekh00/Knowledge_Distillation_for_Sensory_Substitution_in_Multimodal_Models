from tools.audit_app.audit_items import AuditItem, canonicalize_answer
import pytest

from tools.audit_app.audit_store import (
    AuditResponse,
    compute_stats,
    render_report_markdown,
)


def _item(question_id: str, question_type: str, answer: str) -> AuditItem:
    return AuditItem(
        question_id=question_id, image_id="scene/1", question_type=question_type,
        question="Is there a chair?", answer=answer, row={"sensor": "kv2"},
    )


def _response(question_id: str, annotator_id: str, own_answer: str, verdict: str) -> AuditResponse:
    return AuditResponse.new(question_id, annotator_id, own_answer, verdict, notes="")


def test_canonicalize_matches_across_case_number_words_and_underscores():
    assert canonicalize_answer("Three") == canonicalize_answer("3")
    assert canonicalize_answer("night_stand") == canonicalize_answer("Night Stand.")


def test_gold_accuracy_and_ambiguous_share_use_one_reviewer():
    items = [_item("q1", "existence", "yes"), _item("q2", "existence", "no")]
    responses = {
        "ann_a": {"q1": _response("q1", "ann_a", "yes", "correct"),
                  "q2": _response("q2", "ann_a", "no", "ambiguous")},
    }

    [stat] = compute_stats(items, responses)

    assert stat.n_verdicts == 2
    assert stat.gold_accuracy == 0.5
    assert stat.ambiguous_share == 0.5
    assert stat.human_accuracy_vs_gold == 1.0


def test_human_accuracy_ignores_blank_own_answers():
    # Reveal-gold workflow: own_answer is only collected on "incorrect", so
    # a "correct" verdict normally carries an empty own_answer and must not
    # be scored as a wrong "correction" against gold.
    items = [_item("q1", "existence", "yes"), _item("q2", "existence", "no")]
    responses = {
        "ann_a": {
            "q1": _response("q1", "ann_a", "", "correct"),        # no correction offered
            "q2": _response("q2", "ann_a", "yes", "incorrect"),    # correction offered, wrong
        },
    }

    [stat] = compute_stats(items, responses)

    assert stat.human_accuracy_vs_gold == 0.0  # only q2 counts, and its correction ("yes") != gold ("no")


def test_multiple_reviewers_are_rejected_in_single_reviewer_protocol():
    items = [_item("q1", "existence", "yes")]
    responses = {
        "ann_a": {"q1": _response("q1", "ann_a", "yes", "correct")},
        "ann_b": {"q1": _response("q1", "ann_b", "yes", "correct")},
    }

    with pytest.raises(ValueError, match="exactly one reviewer"):
        compute_stats(items, responses)


def test_partial_review_has_no_acceptance_verdict():
    items = [_item("q1", "existence", "yes"), _item("q2", "existence", "no")]
    responses = {"solo": {"q1": _response("q1", "solo", "", "correct")}}

    [stat] = compute_stats(items, responses)

    assert stat.meets_acceptance is None


def test_report_names_single_reviewer_verification_and_omits_kappa():
    items = [_item("q1", "existence", "yes")]
    responses = {"solo": {"q1": _response("q1", "solo", "", "correct")}}
    stats = compute_stats(items, responses)

    report = render_report_markdown(stats, {"solo": "R1"})

    assert "single-reviewer gold verification" in report.lower()
    assert "Cohen" not in report
