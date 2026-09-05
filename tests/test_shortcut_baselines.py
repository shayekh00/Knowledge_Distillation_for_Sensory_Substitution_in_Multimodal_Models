"""Tests for the shortcut baselines (plan §5.3)."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evaluate  # noqa: E402
from evaluation.shortcut_baselines import (  # noqa: E402
    _anchor_key,
    composition_tables,
    shortcut_baselines,
)

COLUMNS = ["question_id", "question_type", "answer", "answer_space", "question",
           "scene_type", "sensor", "variant", "evidence"]


@pytest.fixture(scope="module")
def vocab_tables():
    return (evaluate.load_synonyms(os.path.join(evaluate.VOCAB_DIR, "synonyms.csv")),
            evaluate.load_canonical_vocab(
                os.path.join(evaluate.VOCAB_DIR, "canonical_objects.csv")))


def frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def existence_row(qid, answer, scene_type, concept="table", sensor="kv2"):
    return (qid, "existence", answer, "yes|no", f"Is there any {concept}?",
            scene_type, sensor, "", json.dumps({"concept": concept}))


def test_scene_prior_detects_a_scene_conditioned_answer(vocab_tables):
    synonyms, vocabulary = vocab_tables
    # Globally balanced 50/50, but perfectly separable by scene type.
    train = frame([existence_row(f"t{i}", "no", "bedroom") for i in range(10)]
                  + [existence_row(f"t{i+10}", "yes", "office") for i in range(10)])
    test = frame([existence_row(f"q{i}", "no", "bedroom") for i in range(5)]
                 + [existence_row(f"q{i+5}", "yes", "office") for i in range(5)])
    results = shortcut_baselines(train, test, synonyms, vocabulary)
    # A constant predictor can only get half; the scene label gets all of it.
    assert results["constant"]["existence"] == pytest.approx(0.5)
    assert results["scene_prior"]["existence"] == pytest.approx(1.0)


def test_scene_prior_is_flat_when_the_answer_does_not_depend_on_scene(vocab_tables):
    synonyms, vocabulary = vocab_tables
    train = frame([existence_row(f"t{i}", "yes" if i % 2 else "no", "bedroom")
                   for i in range(10)]
                  + [existence_row(f"t{i+10}", "yes" if i % 2 else "no", "office")
                     for i in range(10)])
    test = frame([existence_row(f"q{i}", "yes" if i % 2 else "no", "bedroom")
                  for i in range(10)])
    results = shortcut_baselines(train, test, synonyms, vocabulary)
    assert results["scene_prior"]["existence"] == pytest.approx(0.5)


def relative_row(qid, answer, space, comparative="closer"):
    a, b = space.split("|")
    return (qid, "relative_depth", answer, space, f"Which is {comparative}?",
            "office", "kv2", comparative,
            json.dumps({"comparative": comparative, "a_concept": a, "b_concept": b}))


def test_answer_position_baselines_are_complementary(vocab_tables):
    synonyms, vocabulary = vocab_tables
    train = frame([relative_row("t1", "chair", "chair|table")])
    test = frame([relative_row("q1", "chair", "chair|table"),
                  relative_row("q2", "table", "chair|table"),
                  relative_row("q3", "chair", "chair|table"),
                  relative_row("q4", "table", "chair|table")])
    results = shortcut_baselines(train, test, synonyms, vocabulary)
    first = results["answer_position_first"]["relative_depth"]
    second = results["answer_position_second"]["relative_depth"]
    assert first == pytest.approx(0.5) and second == pytest.approx(0.5)
    assert first + second == pytest.approx(1.0)


def test_answer_position_is_zero_for_open_vocabulary_types(vocab_tables):
    synonyms, vocabulary = vocab_tables
    rows = [("q1", "nearest_object", "chair", float("nan"), "What is nearest?",
             "office", "kv2", "", json.dumps({"anchor_concept": "bed"}))]
    results = shortcut_baselines(frame(rows), frame(rows), synonyms, vocabulary)
    # No declared answer space, so the baseline cannot predict and scores wrong.
    assert results["answer_position_first"]["nearest_object"] == 0.0


def test_anchor_key_uses_the_generator_evidence():
    row = {"question_type": "nearest_object", "answer_space": float("nan"),
           "variant": "", "evidence": json.dumps({"anchor_concept": "night_stand"})}
    assert _anchor_key("nearest_object", row) == "night_stand"
    row = {"question_type": "existence", "answer_space": "yes|no", "variant": "",
           "evidence": json.dumps({"concept": "table"})}
    assert _anchor_key("existence", row) == "table"


def test_malformed_evidence_does_not_crash():
    row = {"question_type": "existence", "answer_space": "yes|no", "variant": "",
           "evidence": "not json at all"}
    assert _anchor_key("existence", row) is None


def test_baselines_are_fitted_on_train_not_the_evaluated_split(vocab_tables):
    synonyms, vocabulary = vocab_tables
    train = frame([existence_row(f"t{i}", "yes", "office") for i in range(10)])
    test = frame([existence_row(f"q{i}", "no", "office") for i in range(10)])
    results = shortcut_baselines(train, test, synonyms, vocabulary)
    # Fitting on the evaluated split would score 100%; the honest answer is 0%.
    assert results["constant"]["existence"] == 0.0
    assert results["scene_prior"]["existence"] == 0.0


def test_composition_tables_count_every_row(vocab_tables):
    gold = frame([existence_row("q1", "yes", "office", sensor="kv1"),
                  existence_row("q2", "no", "bedroom", sensor="kv2")])
    tables = composition_tables(gold)
    counts = tables["sensor_by_type"]["existence"]
    assert sum(counts.values()) == 2
