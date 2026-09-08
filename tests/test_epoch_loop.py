"""EarlyStopper: the off-by-one surface for protocol §7.3's stopping rule.

"Stop when validation macro accuracy fails to improve over 2 consecutive
evaluation points" is easy to get subtly wrong in two ways: stopping one
epoch too early or late, and reporting the epoch that *triggered* the stop
as the best one rather than the epoch that actually scored highest. The
second failure is silent — training still stops at a sane-looking point,
but the wrong checkpoint gets promoted to the reported adapter.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distillation.epoch_loop import EarlyStopper  # noqa: E402


def run(macros, max_epochs=10, patience=2):
    stopper = EarlyStopper(max_epochs=max_epochs, patience=patience)
    stopped_at = None
    for epoch, macro in enumerate(macros):
        if stopper.step(epoch, macro):
            stopped_at = epoch
            break
    return stopper, stopped_at


def test_stops_after_patience_consecutive_non_improvements():
    # improves at 0, 1; flat at 2, 3 -> patience=2 exhausted at epoch 3.
    stopper, stopped_at = run([0.40, 0.45, 0.45, 0.45], patience=2)
    assert stopped_at == 3
    assert stopper.best_epoch == 1
    assert stopper.best_macro == pytest.approx(0.45)


def test_best_epoch_is_the_peak_not_the_epoch_that_triggered_the_stop():
    """The dangerous case: accuracy peaks, then declines for `patience` epochs.
    The stopper must not report the last (worst) epoch as best."""
    stopper, stopped_at = run([0.30, 0.50, 0.40, 0.35], patience=2)
    assert stopped_at == 3
    assert stopper.best_epoch == 1
    assert stopper.best_macro == pytest.approx(0.50)


def test_an_improvement_resets_the_patience_counter():
    # dip at 1, then a new best at 2 must reset the clock, not just note it.
    stopper, stopped_at = run([0.40, 0.35, 0.50, 0.45, 0.45], patience=2)
    assert stopped_at == 4
    assert stopper.best_epoch == 2


def test_stops_at_max_epochs_even_while_still_improving():
    stopper, stopped_at = run([0.10, 0.20, 0.30, 0.40], max_epochs=4, patience=10)
    assert stopped_at == 3
    assert stopper.best_epoch == 3


def test_a_tie_does_not_count_as_improvement():
    """Equal, not greater, must not reset patience — otherwise a model stuck
    on a plateau never stops within budget."""
    stopper, stopped_at = run([0.40, 0.40, 0.40], patience=2)
    assert stopped_at == 2
    assert stopper.best_epoch == 0


def test_single_epoch_budget_stops_immediately_and_is_its_own_best():
    stopper, stopped_at = run([0.40], max_epochs=1, patience=2)
    assert stopped_at == 0
    assert stopper.best_epoch == 0


def test_history_records_every_epoch_seen():
    stopper, _ = run([0.40, 0.45, 0.45], patience=2)
    assert [entry["epoch"] for entry in stopper.history] == [0, 1, 2]
    assert [entry["val_macro"] for entry in stopper.history] == pytest.approx([0.40, 0.45, 0.45])


def test_rejects_nonsensical_construction():
    with pytest.raises(ValueError, match="max_epochs"):
        EarlyStopper(max_epochs=0, patience=2)
    with pytest.raises(ValueError, match="patience"):
        EarlyStopper(max_epochs=10, patience=0)


def test_partial_coverage_is_announced_not_silently_diluted(capsys, monkeypatch):
    """The artifact that produced three wrong readings: a partial prediction set
    scored against the whole split, which looks like failure rather than
    coverage. It stays permitted (smoke tests need it) but must be loud.
    """
    import pandas as pd
    import distillation.epoch_loop as epoch_loop

    gold = pd.DataFrame({"question_id": [f"q{i}" for i in range(100)],
                         "answer": ["yes"] * 100})
    monkeypatch.setattr(epoch_loop, "load_release_split", lambda *a, **k: gold)
    monkeypatch.setattr(epoch_loop, "load_synonyms", lambda *a, **k: {})
    monkeypatch.setattr(epoch_loop, "load_canonical_vocab", lambda *a, **k: set())
    monkeypatch.setattr(epoch_loop, "score_predictions", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(epoch_loop, "macro_accuracy", lambda *a, **k: 0.0)

    epoch_loop.score_val_macro([("q0", "yes")], split="val")
    output = capsys.readouterr().out
    assert "PARTIAL COVERAGE" in output
    assert "1 predictions" in output and "100 gold rows" in output
    assert "must" in output and "not be quoted" in output


def test_full_coverage_is_not_flagged(capsys, monkeypatch):
    import pandas as pd
    import distillation.epoch_loop as epoch_loop

    gold = pd.DataFrame({"question_id": ["q0", "q1"], "answer": ["yes", "no"]})
    monkeypatch.setattr(epoch_loop, "load_release_split", lambda *a, **k: gold)
    monkeypatch.setattr(epoch_loop, "load_synonyms", lambda *a, **k: {})
    monkeypatch.setattr(epoch_loop, "load_canonical_vocab", lambda *a, **k: set())
    monkeypatch.setattr(epoch_loop, "score_predictions", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(epoch_loop, "macro_accuracy", lambda *a, **k: 0.5)

    epoch_loop.score_val_macro([("q0", "yes"), ("q1", "no")], split="val")
    assert "PARTIAL COVERAGE" not in capsys.readouterr().out
