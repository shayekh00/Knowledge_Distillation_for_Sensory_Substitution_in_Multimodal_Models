from tools.audit_app.audit_items import AuditItem, canonicalize_answer
from tools.audit_app.audit_store import AuditResponse, compute_stats


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


def test_gold_accuracy_and_ambiguous_share_pool_across_annotators():
    items = [_item("q1", "existence", "yes"), _item("q2", "existence", "no")]
    responses = {
        "ann_a": {"q1": _response("q1", "ann_a", "yes", "correct"),
                  "q2": _response("q2", "ann_a", "no", "ambiguous")},
        "ann_b": {"q1": _response("q1", "ann_b", "yes", "correct"),
                  "q2": _response("q2", "ann_b", "yes", "incorrect")},
    }

    [stat] = compute_stats(items, responses)

    assert stat.n_verdicts == 4
    assert stat.gold_accuracy == 0.5   # 2 of 4 verdicts are "correct"
    assert stat.ambiguous_share == 0.25
    assert stat.human_accuracy_vs_gold == 0.75  # 3 of 4 own answers match gold


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


def test_kappa_is_none_without_exactly_two_annotators():
    items = [_item("q1", "existence", "yes")]
    only_one = {"ann_a": {"q1": _response("q1", "ann_a", "yes", "correct")}}

    [stat] = compute_stats(items, only_one)

    assert stat.cohen_kappa is None
    assert "2 annotators" in stat.kappa_note


def test_kappa_computed_when_two_annotators_reviewed_the_same_items():
    items = [_item(f"q{i}", "existence", "yes") for i in range(4)]
    responses = {
        "ann_a": {item.question_id: _response(item.question_id, "ann_a", "yes", verdict)
                  for item, verdict in zip(items, ["correct", "correct", "incorrect", "ambiguous"])},
        "ann_b": {item.question_id: _response(item.question_id, "ann_b", "yes", verdict)
                  for item, verdict in zip(items, ["correct", "incorrect", "incorrect", "ambiguous"])},
    }

    [stat] = compute_stats(items, responses)

    assert stat.cohen_kappa is not None
    assert -1.0 <= stat.cohen_kappa <= 1.0
