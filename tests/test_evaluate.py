"""Tests for the shipped evaluation protocol (§9)."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    macro_accuracy,
    majority_baseline,
    score_predictions,
    snap_to_answer_space,
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
    assert snap_to_answer_space("It is on the left.", "left|right", "left_right",
                                synonyms, vocabulary) == "left"
    # Nothing matches -> forced to commit to a legal option, never to abstain.
    assert snap_to_answer_space("banana", "left|right", "left_right",
                                synonyms, vocabulary) == "left"


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
