"""Tests for the shipped evaluation protocol (§9)."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    check_reported_aggregates,
    is_missing,
    load_predictions,
    macro_accuracy,
    majority_baseline,
    parse_answer_space,
    prediction_id_report,
    random_baseline,
    score_predictions,
    snap_to_answer_space,
    split_majority_share,
    theoretical_chance,
)


@pytest.fixture(scope="module")
def vocab_tables():
    return (evaluate.load_synonyms(os.path.join(evaluate.VOCAB_DIR, "synonyms.csv")),
            evaluate.load_canonical_vocab(os.path.join(evaluate.VOCAB_DIR, "canonical_objects.csv")))


def gold_frame(rows):
    return pd.DataFrame(rows, columns=["question_id", "question_type", "answer",
                                       "answer_space", "question"])


def predictions_frame(pairs):
    return pd.DataFrame(pairs, columns=["question_id", "prediction"])


def test_synonym_and_surface_form_count_as_correct(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "nearest_object", "sofa", None, "?"),
                       ("q2", "nearest_object", "table", None, "?")])
    predictions = predictions_frame([("q1", "couch"), ("q2", "The table.")])
    [score] = score_predictions(gold, predictions, synonyms, vocabulary)
    assert score.accuracy == 1.0


def test_missing_prediction_counts_as_wrong_not_dropped(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "existence", "yes", "yes|no", "?"),
                       ("q2", "existence", "no", "yes|no", "?")])
    [score] = score_predictions(gold, predictions_frame([("q1", "yes")]), synonyms, vocabulary)
    assert score.n_items == 2 and score.n_predicted == 1
    assert score.accuracy == 0.5


def test_blank_prediction_is_not_a_free_pass(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "existence", "yes", "yes|no", "?")])
    [score] = score_predictions(gold, predictions_frame([("q1", "   ")]), synonyms, vocabulary)
    assert score.accuracy == 0.0


def test_constrained_decoding_snaps_onto_the_answer_space(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # Trailing punctuation must not defeat the mention search.
    assert snap_to_answer_space("It is on the left.", "left|right", "left_right",
                                synonyms, vocabulary) == "left"
    assert snap_to_answer_space("Right!", "left|right", "left_right",
                                synonyms, vocabulary) == "right"


def test_unparsable_answer_is_not_snapped_to_the_first_option(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # An answer naming no option is left alone, and so scores wrong. Snapping it
    # onto options[0] would invent a decision the model never made and hand out
    # chance accuracy on every binary type (plan §6.3).
    assert snap_to_answer_space("banana", "left|right", "left_right",
                                synonyms, vocabulary) == "banana"
    gold = gold_frame([("q1", "left_right", "left", "left|right", "?")])
    predictions = predictions_frame([("q1", "banana")])
    [score] = score_predictions(gold, predictions, synonyms, vocabulary, constrained=True)
    assert score.accuracy == 0.0
    assert score.n_invalid == 1


def test_constrained_scoring_can_rescue_a_verbose_answer(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "left_right", "right", "left|right", "?")])
    predictions = predictions_frame([("q1", "the right side")])
    assert score_predictions(gold, predictions, synonyms, vocabulary)[0].accuracy == 0.0
    assert score_predictions(gold, predictions, synonyms, vocabulary,
                             constrained=True)[0].accuracy == 1.0


def test_relative_depth_macro_f1_uses_answer_position(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "relative_depth", "chair", "chair|table", "?"),
                       ("q2", "relative_depth", "table", "chair|table", "?")])
    predictions = predictions_frame([("q1", "chair"), ("q2", "table")])
    [score] = score_predictions(gold, predictions, synonyms, vocabulary)
    assert score.accuracy == 1.0
    assert score.macro_f1 == pytest.approx(1.0)


def test_open_vocabulary_types_report_no_macro_f1(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "nearest_object", "chair", None, "?")])
    [score] = score_predictions(gold, predictions_frame([("q1", "chair")]), synonyms, vocabulary)
    assert score.macro_f1 is None


def test_macro_accuracy_is_unweighted_across_types(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "existence", "yes", "yes|no", "?"),
                       ("q2", "existence", "yes", "yes|no", "?"),
                       ("q3", "existence", "yes", "yes|no", "?"),
                       ("q4", "left_right", "left", "left|right", "?")])
    predictions = predictions_frame([("q1", "yes"), ("q2", "yes"), ("q3", "yes"),
                                     ("q4", "right")])
    scores = score_predictions(gold, predictions, synonyms, vocabulary)
    # 3/4 items right, but one of two types scored zero: macro is 50%, not 75%.
    assert macro_accuracy(scores) == pytest.approx(0.5)


def test_majority_baseline_reads_the_train_split_not_the_evaluated_one():
    train = gold_frame([("t1", "existence", "yes", "yes|no", "?"),
                        ("t2", "existence", "yes", "yes|no", "?")])
    evaluated = gold_frame([("q1", "existence", "no", "yes|no", "?"),
                            ("q2", "existence", "no", "yes|no", "?")])
    # An oracle majority on the evaluated split would score 100%; the train
    # majority is "yes", so the honest baseline is 0%.
    assert majority_baseline(train, evaluated)["existence"] == 0.0


def test_snapping_prefers_the_option_mentioned_first(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # Both options appear; the one the model actually chose is named first.
    assert snap_to_answer_space("the chair is closer than the table", "chair|table",
                                "relative_depth", synonyms, vocabulary) == "chair"
    assert snap_to_answer_space("the table is closer than the chair", "chair|table",
                                "relative_depth", synonyms, vocabulary) == "table"


def test_open_vocabulary_predictions_are_not_snapped(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # Open types declare no answer_space; NaN must not be read as an option.
    for empty in (None, float("nan"), ""):
        assert snap_to_answer_space("bookshelf", empty, "nearest_object",
                                    synonyms, vocabulary) == "bookshelf"


def test_constrained_scoring_leaves_open_types_intact(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "nearest_object", "chair", float("nan"), "?")])
    predictions = predictions_frame([("q1", "chair")])
    [score] = score_predictions(gold, predictions, synonyms, vocabulary, constrained=True)
    assert score.accuracy == 1.0


# --------------------------------------------------------------------------
# Score fixtures and edge cases (plan §6.2 / §19 Phase 3).
# --------------------------------------------------------------------------


def test_nan_answer_space_is_not_read_as_the_option_nan():
    # NaN is truthy, so `answer_space or ""` yields the string "nan" and every
    # caller then treats "nan" as the row's only legal answer. This is the
    # defect that pinned both open-vocabulary random baselines to 0.0%.
    for empty in (None, float("nan"), "", "   ", "nan"):
        assert parse_answer_space(empty) == []
    assert parse_answer_space("left|right") == ["left", "right"]
    assert parse_answer_space(" chair | tissue box ") == ["chair", "tissue box"]


def test_missing_detects_nan_before_stringifying():
    assert is_missing(None) and is_missing(float("nan")) and is_missing("  ")
    assert not is_missing("no") and not is_missing("0")


def test_nan_prediction_is_absent_not_answered(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # str(NaN) == "nan", which is non-empty: a NaN prediction used to count as
    # an item the model successfully answered.
    gold = gold_frame([("q1", "existence", "yes", "yes|no", "?")])
    [score] = score_predictions(gold, predictions_frame([("q1", float("nan"))]),
                                synonyms, vocabulary)
    assert score.n_predicted == 0
    assert score.accuracy == 0.0


def test_random_baseline_uses_the_vocabulary_for_open_types(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([(f"q{i}", "nearest_object", "chair", float("nan"), "?")
                       for i in range(400)])
    accuracy = random_baseline(gold, synonyms, vocabulary)["nearest_object"]
    # Guessing uniformly over ~151 concepts must land near 1/151, not at zero.
    assert 0.0 < accuracy < 0.05


def test_relative_depth_chance_is_one_half_not_one_over_the_vocabulary(vocab_tables):
    _, vocabulary = vocab_tables
    gold = gold_frame([("q1", "relative_depth", "chair", "chair|table", "?"),
                       ("q2", "relative_depth", "table", "chair|table", "?")])
    assert theoretical_chance(gold, vocabulary)["relative_depth"] == pytest.approx(0.5)


def test_split_majority_share_is_reported_apart_from_the_train_majority():
    train = gold_frame([("t1", "existence", "yes", "yes|no", "?"),
                        ("t2", "existence", "yes", "yes|no", "?")])
    evaluated = gold_frame([("q1", "existence", "no", "yes|no", "?"),
                            ("q2", "existence", "no", "yes|no", "?")])
    # The honest train-fitted baseline scores 0%; the split's own majority share
    # is 100%. They must never be printed as the same quantity.
    assert majority_baseline(train, evaluated)["existence"] == 0.0
    assert split_majority_share(evaluated)["existence"] == 1.0


def test_duplicate_prediction_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicated.csv"
    path.write_text("question_id,prediction\nq1,yes\nq1,no\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicated question_id"):
        load_predictions(str(path))


def test_unexpected_prediction_ids_are_reported(tmp_path):
    path = tmp_path / "extra.csv"
    path.write_text("question_id,prediction\nq1,yes\nqZ,no\n", encoding="utf-8")
    gold = gold_frame([("q1", "existence", "yes", "yes|no", "?"),
                       ("q2", "existence", "no", "yes|no", "?")])
    report = prediction_id_report(gold, load_predictions(str(path)))
    assert report["n_unexpected"] == 1 and report["unexpected_examples"] == ["qZ"]
    assert report["n_missing"] == 1


def scoring_fixture():
    """One row per type, covering both binary polarities and a multiword object."""
    return gold_frame([
        ("f1", "existence", "yes", "yes|no", "Is there a chair?"),
        ("f2", "existence", "no", "yes|no", "Is there a lamp?"),
        ("f3", "left_right", "left", "left|right", "Left or right?"),
        ("f4", "left_right", "right", "left|right", "Left or right?"),
        ("f5", "relative_depth", "chair", "chair|table", "Which is closer?"),
        ("f6", "nearest_object", "tissue box", float("nan"), "What is nearest?"),
    ])


def test_gold_predictions_score_one_hundred_percent(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = scoring_fixture()
    predictions = predictions_frame(list(zip(gold["question_id"], gold["answer"])))
    scores = score_predictions(gold, predictions, synonyms, vocabulary)
    assert macro_accuracy(scores) == pytest.approx(1.0)
    assert sum(score.n_invalid for score in scores) == 0


def test_all_empty_predictions_score_zero(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = scoring_fixture()
    predictions = predictions_frame([(qid, "") for qid in gold["question_id"]])
    scores = score_predictions(gold, predictions, synonyms, vocabulary)
    assert macro_accuracy(scores) == 0.0
    assert sum(score.n_predicted for score in scores) == 0


def test_antonyms_never_match(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = scoring_fixture()
    flipped = {"yes": "no", "no": "yes", "left": "right", "right": "left"}
    predictions = predictions_frame([
        (row["question_id"], flipped.get(row["answer"], row["answer"]))
        for _, row in gold.iterrows() if row["answer"] in flipped])
    scores = score_predictions(gold[gold["answer"].isin(flipped)], predictions,
                               synonyms, vocabulary)
    assert all(score.accuracy == 0.0 for score in scores)


def test_multiword_object_answer_survives_surface_variation(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "nearest_object", "tissue box", float("nan"), "?")])
    for surface in ("tissue box", "The tissue box.", "tissue_box", "  Tissue Box  "):
        [score] = score_predictions(gold, predictions_frame([("q1", surface)]),
                                    synonyms, vocabulary)
        assert score.accuracy == 1.0, surface


def test_relative_depth_answer_order_is_respected(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # Same two objects, opposite gold answers: a model that always says the
    # first-named option must score 50%, not 100%.
    gold = gold_frame([("q1", "relative_depth", "chair", "chair|table", "?"),
                       ("q2", "relative_depth", "table", "chair|table", "?")])
    predictions = predictions_frame([("q1", "chair"), ("q2", "chair")])
    [score] = score_predictions(gold, predictions, synonyms, vocabulary)
    assert score.accuracy == pytest.approx(0.5)


def test_invalid_answers_do_not_mint_new_f1_classes(vocab_tables):
    synonyms, vocabulary = vocab_tables
    gold = gold_frame([("q1", "existence", "yes", "yes|no", "?"),
                       ("q2", "existence", "no", "yes|no", "?")])
    # "banana" must not become a third F1 class; it is a false negative for "no".
    predictions = predictions_frame([("q1", "yes"), ("q2", "banana")])
    [score] = score_predictions(gold, predictions, synonyms, vocabulary)
    assert score.n_invalid == 1
    assert score.macro_f1 == pytest.approx(f1_score(["yes", "no"], ["yes", "<invalid>"],
                                                    labels=["no", "yes"], average="macro",
                                                    zero_division=0))


def test_reported_macro_must_equal_the_per_type_mean(vocab_tables):
    synonyms, vocabulary = vocab_tables
    scores = score_predictions(scoring_fixture(),
                               predictions_frame([("f1", "yes")]), synonyms, vocabulary)
    check_reported_aggregates(scores, macro_accuracy(scores))
    with pytest.raises(SystemExit, match="Aggregate check failed"):
        check_reported_aggregates(scores, macro_accuracy(scores) + 0.01)
