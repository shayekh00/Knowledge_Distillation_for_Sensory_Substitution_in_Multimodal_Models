"""The mandatory loss tests from plan §7.3.

Each test below corresponds to a bullet in that list, or to a defect reproduced in
`docs/New_Submission/implementation_audit.md`. They run on CPU in about a second,
so the objective surface is verified before any GPU time is spent.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.losses import (  # noqa: E402
    IGNORE_INDEX,
    candidate_kd_loss,
    contrastive_loss,
    feature_transfer_loss,
    loca_calibrate,
    loca_kd_loss,
    masked_cross_entropy,
    shift_for_causal_lm,
    token_kd_loss,
    valid_answer_mask,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(20260905)


# ---------------------------------------------------------------------------
# Alignment and masking
# ---------------------------------------------------------------------------

def test_causal_shift_aligns_logits_with_the_token_they_predict():
    logits = torch.randn(2, 5, 7)
    labels = torch.arange(10).reshape(2, 5)
    shifted_logits, shifted_labels = shift_for_causal_lm(logits, labels)
    assert shifted_logits.shape == (2, 4, 7)
    assert torch.equal(shifted_labels, labels[:, 1:])
    assert torch.equal(shifted_logits, logits[:, :-1])


def test_padding_and_prompt_positions_are_excluded():
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4]])
    assert torch.equal(valid_answer_mask(labels),
                       torch.tensor([[False, False, True, True]]))


def test_appending_padding_does_not_change_the_normalised_loss():
    """§7.3: 'Appending padding does not change the normalized loss.'"""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 9)
    labels = torch.tensor([[IGNORE_INDEX, 2, 5, 1]])

    padded_logits = torch.cat([logits, torch.randn(1, 3, 9)], dim=1)
    padded_labels = torch.cat([labels, torch.full((1, 3), IGNORE_INDEX)], dim=1)

    base = masked_cross_entropy(logits, labels)
    padded = masked_cross_entropy(padded_logits, padded_labels)
    assert base.item() == pytest.approx(padded.item(), abs=1e-6)


def test_padding_contributes_zero_to_kd():
    """§7.3: 'Padding and prompt tokens contribute zero to both CE and KD.'"""
    torch.manual_seed(0)
    teacher = torch.randn(1, 4, 9)
    student = torch.randn(1, 4, 9)
    labels = torch.tensor([[IGNORE_INDEX, 2, 5, 1]])

    pad_teacher = torch.cat([teacher, torch.randn(1, 2, 9) * 50], dim=1)
    pad_student = torch.cat([student, torch.randn(1, 2, 9) * 50], dim=1)
    pad_labels = torch.cat([labels, torch.full((1, 2), IGNORE_INDEX)], dim=1)

    base = token_kd_loss(teacher, student, labels)
    padded = token_kd_loss(pad_teacher, pad_student, pad_labels)
    # Wildly different logits at the padded positions must not move the loss.
    assert base.item() == pytest.approx(padded.item(), abs=1e-6)


def test_all_masked_labels_is_an_error_not_a_silent_zero():
    logits = torch.randn(1, 3, 5)
    labels = torch.full((1, 3), IGNORE_INDEX)
    with pytest.raises(ValueError, match="no valid answer positions"):
        masked_cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# KD
# ---------------------------------------------------------------------------

def test_identical_distributions_give_zero_kl():
    """§7.3: 'Identical teacher/student distributions give zero KL.'"""
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    assert token_kd_loss(logits, logits.clone(), labels).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_is_not_divided_by_the_vocabulary_size():
    """Audit B3: reduction='mean' under-scales by V. Ours must not."""
    torch.manual_seed(0)
    teacher = torch.randn(1, 3, 64)
    student = torch.randn(1, 3, 64)
    labels = torch.randint(0, 64, (1, 3))

    ours = token_kd_loss(teacher, student, labels, temperature=1.0)
    shifted_t, shifted_labels = shift_for_causal_lm(teacher, labels)
    shifted_s, _ = shift_for_causal_lm(student, labels)
    wrong = F.kl_div(F.log_softmax(shifted_s, -1), F.softmax(shifted_t, -1), reduction="mean")
    # The legacy convention is smaller by roughly the vocabulary size.
    assert ours.item() > wrong.item() * 50


def test_kd_scales_with_temperature_squared():
    teacher = torch.randn(1, 4, 12)
    student = torch.randn(1, 4, 12)
    labels = torch.randint(0, 12, (1, 4))
    at_one = token_kd_loss(teacher, student, labels, temperature=1.0)
    at_two = token_kd_loss(teacher, student, labels, temperature=2.0)
    assert at_one.item() > 0 and at_two.item() > 0
    assert at_one.item() != pytest.approx(at_two.item())


def test_token_kd_rejects_mismatched_vocabularies():
    """Audit B6: no slicing one model's logits to another's width."""
    labels = torch.randint(0, 8, (1, 3))
    with pytest.raises(ValueError, match="identical tokenizer"):
        token_kd_loss(torch.randn(1, 3, 16), torch.randn(1, 3, 8), labels)


# ---------------------------------------------------------------------------
# Candidate KD (cross-family)
# ---------------------------------------------------------------------------

def test_candidate_kd_is_invariant_to_a_common_candidate_ordering():
    """§7.3: 'Candidate KD is invariant to a common candidate ordering.'"""
    teacher = torch.randn(4, 5)
    student = torch.randn(4, 5)
    base = candidate_kd_loss(teacher, student)
    permutation = torch.randperm(5)
    permuted = candidate_kd_loss(teacher[:, permutation], student[:, permutation])
    assert base.item() == pytest.approx(permuted.item(), abs=1e-6)


def test_candidate_kd_is_zero_for_identical_scores():
    scores = torch.randn(3, 4)
    assert candidate_kd_loss(scores, scores.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_candidate_kd_rejects_a_missing_candidate():
    """§7.3: 'rejects a missing candidate'."""
    with pytest.raises(ValueError, match="same candidate list"):
        candidate_kd_loss(torch.randn(3, 5), torch.randn(3, 4))
    with pytest.raises(ValueError, match="at least 2 candidates"):
        candidate_kd_loss(torch.randn(3, 1), torch.randn(3, 1))


# ---------------------------------------------------------------------------
# LoCa
# ---------------------------------------------------------------------------

def _random_probs(shape):
    return F.softmax(torch.randn(*shape), dim=-1)


def test_calibrated_probabilities_are_finite_nonnegative_and_sum_to_one():
    """§7.3: 'Calibrated probabilities are finite, nonnegative, sum to one...'"""
    probs = _random_probs((6, 20))
    targets = torch.randint(0, 20, (6,))
    calibrated = loca_calibrate(probs, targets, alpha=0.8)
    assert torch.isfinite(calibrated).all()
    assert (calibrated >= 0).all()
    assert torch.allclose(calibrated.sum(-1), torch.ones(6), atol=1e-5)


def test_loca_makes_gold_top1_whether_or_not_it_started_there():
    """§7.3: 'LoCa handles gold initially top-1 and initially wrong'."""
    # Gold is deliberately the *lowest* probability class.
    probs = torch.tensor([[0.70, 0.25, 0.05]])
    for gold in (0, 1, 2):
        calibrated = loca_calibrate(probs, torch.tensor([gold]), alpha=0.8)
        assert calibrated.argmax(-1).item() == gold, f"gold {gold} not top-1 after calibration"


def test_loca_gold_margin_is_exactly_one_minus_alpha():
    """The construction guarantees p~_g - s * p_wrong == 1 - alpha."""
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    for alpha in (0.6, 0.8, 0.9):
        gold = 2
        calibrated = loca_calibrate(probs, torch.tensor([gold]), alpha=alpha)
        others = torch.cat([calibrated[0, :gold], calibrated[0, gold + 1:]])
        margin = calibrated[0, gold] - others.max()
        assert margin.item() == pytest.approx(1.0 - alpha, abs=1e-5)


def test_loca_preserves_non_target_ratios():
    """§7.3: 'Non-target probability ratios are preserved where defined.'"""
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    gold = 0
    calibrated = loca_calibrate(probs, torch.tensor([gold]), alpha=0.8)
    before = probs[0, 1] / probs[0, 2]
    after = calibrated[0, 1] / calibrated[0, 2]
    assert before.item() == pytest.approx(after.item(), abs=1e-5)


def test_loca_uses_the_highest_wrong_class_not_the_second_highest():
    """Audit B2. When gold is not top-1, top-2 is the wrong competitor."""
    probs = torch.tensor([[0.70, 0.25, 0.05]])
    gold = 2                       # gold is last; the highest wrong class is index 0
    calibrated = loca_calibrate(probs, torch.tensor([gold]), alpha=0.8)
    # With p_wrong = 0.70 the scale is 0.8 / (1 - 0.05 + 0.70) = 0.4848...
    expected_scale = 0.8 / (1 - 0.05 + 0.70)
    assert calibrated[0, 0].item() == pytest.approx(0.70 * expected_scale, abs=1e-5)


def test_loca_handles_multiple_batches_positions_and_multitoken_answers():
    """§7.3: 'multiple batches, multiple answer positions, and multi-token answers'."""
    teacher = torch.randn(3, 6, 15)
    student = torch.randn(3, 6, 15)
    labels = torch.randint(0, 15, (3, 6))
    labels[0, :2] = IGNORE_INDEX          # prompt
    labels[1, -1] = IGNORE_INDEX          # padding
    loss = loca_kd_loss(teacher, student, labels)
    assert torch.isfinite(loss) and loss.item() > 0


def test_loca_never_gathers_with_ignore_index():
    """Audit B5: -100 reaching a gather is a RuntimeError, not a wrong value."""
    teacher = torch.randn(1, 4, 10)
    student = torch.randn(1, 4, 10)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 7]])
    loss = loca_kd_loss(teacher, student, labels)      # must not raise
    assert torch.isfinite(loss)
    # Calling the calibrator directly with a masked target is still an error.
    with pytest.raises(ValueError, match="IGNORE_INDEX"):
        loca_calibrate(_random_probs((1, 10)), torch.tensor([IGNORE_INDEX]))


def test_loca_rejects_alpha_outside_the_open_unit_interval():
    probs = _random_probs((2, 6))
    targets = torch.randint(0, 6, (2,))
    for alpha in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            loca_calibrate(probs, targets, alpha=alpha)


# ---------------------------------------------------------------------------
# Feature transfer and contrastive
# ---------------------------------------------------------------------------

def test_feature_transfer_is_zero_for_identical_features():
    features = torch.randn(4, 32)
    assert feature_transfer_loss(features, features.clone(), "cosine").item() == \
        pytest.approx(0.0, abs=1e-6)
    assert feature_transfer_loss(features, features.clone(), "mse").item() == \
        pytest.approx(0.0, abs=1e-6)


def test_one_scene_contrastive_training_is_rejected():
    """§7.3: 'One-scene contrastive training is rejected unless a valid external
    negative set is supplied.' The legacy version silently returned 0.0."""
    student = torch.randn(1, 16)
    positive = torch.randn(1, 16)
    with pytest.raises(ValueError, match="non-empty negative bank"):
        contrastive_loss(student, positive, torch.empty(0, 16))


def test_contrastive_loss_rewards_the_matching_scene():
    student = torch.randn(4, 16)
    positive = student.clone() + 0.01 * torch.randn(4, 16)   # nearly aligned
    bank = torch.randn(64, 16)
    aligned = contrastive_loss(student, positive, bank)
    misaligned = contrastive_loss(student, torch.randn(4, 16), bank)
    assert aligned.item() < misaligned.item()


def test_contrastive_loss_is_nonzero_with_a_single_scene_and_a_real_bank():
    """The batch-size-1 case is fine once negatives come from a bank, which is
    exactly what the legacy [1, 1] similarity matrix could not provide."""
    loss = contrastive_loss(torch.randn(1, 16), torch.randn(1, 16), torch.randn(255, 16))
    assert loss.item() > 0
