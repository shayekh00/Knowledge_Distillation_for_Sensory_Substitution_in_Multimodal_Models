"""Distillation losses as pure tensor functions (plan §7.2, §7.3).

Nothing here touches a model, a processor, or a checkpoint. Every function takes
tensors and returns tensors, so the whole objective surface is testable on CPU in
under a second and the GPU window is spent on profiling and training rather than
on debugging tensor math.

Each function encodes a requirement from `docs/New_Submission/implementation_audit.md`,
where the corresponding defect in the legacy path is documented and reproduced:

* **A3** — losses act only on valid answer positions, after a causal shift.
* **B1** — LoCa uses explicit `gather`/`scatter`, never `x[:, :, labels]`.
* **B2** — the non-target class is `argmax over c != gold`, not "second highest".
* **B3** — KL sums over the vocabulary and averages over valid positions, times T².
* **B4** — a contrastive batch with no negatives is an error, not a zero loss.
* **B5** — `IGNORE_INDEX` never reaches a `gather`.
* **B6** — no cross-tokenizer vocabulary slicing; use `candidate_kd_loss` instead.

Probability and KL operations run in float32 even under bf16 autocast, because
the calibration in `loca_calibrate` divides by a difference of probabilities.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Position selection
# ---------------------------------------------------------------------------

def shift_for_causal_lm(logits: torch.Tensor, labels: torch.Tensor):
    """Align next-token logits with the tokens they predict.

    A causal LM's logits at position *t* predict the token at *t+1*. Comparing
    `logits[t]` against `labels[t]` — as the legacy LoCa did — trains and
    distills against an off-by-one target.

    Args:
        logits: ``[B, L, V]``
        labels: ``[B, L]``
    Returns:
        ``(logits[:, :-1], labels[:, 1:])``, both length ``L-1``.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, L, V], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} do not match logits {tuple(logits.shape[:2])}")
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def valid_answer_mask(labels: torch.Tensor) -> torch.Tensor:
    """``True`` where a label is a real supervised target.

    Everything the collator masked — padding, and (once masked correctly) the
    prompt and image placeholders — is excluded. Every loss below is averaged
    over exactly these positions, so appending padding cannot change a
    normalized loss.
    """
    return labels != IGNORE_INDEX


def _safe_targets(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Labels with masked entries replaced by 0, safe to `gather` with.

    `gather` raises on ``-100`` (reproduced in the audit as a hard RuntimeError),
    so masked positions get a dummy index whose contribution is discarded by the
    mask afterwards. The dummy is never allowed to influence a returned value.
    """
    return torch.where(mask, labels, torch.zeros_like(labels))


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------

def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                         already_shifted: bool = False) -> torch.Tensor:
    """Mean CE over valid answer positions only.

    Args:
        logits: ``[B, L, V]``
        labels: ``[B, L]`` with ``IGNORE_INDEX`` where no target applies.
        already_shifted: set when the caller has applied the causal shift.
    """
    if not already_shifted:
        logits, labels = shift_for_causal_lm(logits, labels)
    mask = valid_answer_mask(labels)
    if not mask.any():
        raise ValueError("no valid answer positions: every label is IGNORE_INDEX")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX)


# ---------------------------------------------------------------------------
# Output distillation
# ---------------------------------------------------------------------------

def _kl_over_valid(teacher_probs: torch.Tensor, student_log_probs: torch.Tensor,
                   mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sum KL over the vocabulary, average over valid positions, scale by T².

    `F.kl_div(..., reduction='mean')` instead divides by ``B*L*V``, which
    under-scales the term by the vocabulary size — a factor of ~152k on the real
    tokenizer, which is why the audit records it as effectively switching KD off.
    """
    per_position = (teacher_probs * (teacher_probs.clamp_min(1e-12).log() - student_log_probs)
                    ).sum(dim=-1)
    return (per_position * mask).sum() / mask.sum().clamp_min(1) * (temperature ** 2)


def token_kd_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                  labels: torch.Tensor, temperature: float = 1.0,
                  already_shifted: bool = False) -> torch.Tensor:
    """Token-level KL between teacher and student next-token distributions.

    Only valid for a **verified identical tokenizer**: it compares vocabulary
    index to vocabulary index. For any cross-family pair use
    :func:`candidate_kd_loss` — slicing one model's logits to another's width
    assumes an index correspondence that does not exist.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher {tuple(teacher_logits.shape)} and student "
            f"{tuple(student_logits.shape)} logits differ. Token-level KD requires an "
            "identical tokenizer; use candidate_kd_loss for cross-family pairs.")
    if not already_shifted:
        teacher_logits, _ = shift_for_causal_lm(teacher_logits, labels)
        student_logits, labels = shift_for_causal_lm(student_logits, labels)
    mask = valid_answer_mask(labels)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return _kl_over_valid(teacher_probs, student_log_probs, mask, temperature)


def candidate_kd_loss(teacher_scores: torch.Tensor, student_scores: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
    """KL over the legal-answer candidate distribution (cross-family KD).

    Each model scores the *same* predeclared candidate list with its own
    tokenizer and prompt; the scores are normalized into a distribution over
    candidates and compared. Nothing here assumes shared vocabulary indices,
    hidden sizes, or visual token grids.

    Args:
        teacher_scores: ``[N, C]`` log-scores, one row per question.
        student_scores: ``[N, C]`` log-scores in the **same candidate order**.
    """
    if teacher_scores.shape != student_scores.shape:
        raise ValueError(
            f"candidate score shapes differ: teacher {tuple(teacher_scores.shape)} vs "
            f"student {tuple(student_scores.shape)}. Both models must score the same "
            "candidate list in the same order.")
    if teacher_scores.dim() != 2 or teacher_scores.size(-1) < 2:
        raise ValueError("candidate scores must be [N, C] with at least 2 candidates")
    teacher_probs = F.softmax(teacher_scores.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_scores.float() / temperature, dim=-1)
    per_row = (teacher_probs * (teacher_probs.clamp_min(1e-12).log() - student_log_probs)).sum(-1)
    return per_row.mean() * (temperature ** 2)


# ---------------------------------------------------------------------------
# LoCa (label-conditioned calibration, ECAI 2024)
# ---------------------------------------------------------------------------

def loca_calibrate(teacher_probs: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 0.8) -> torch.Tensor:
    """Rescale a teacher distribution so the gold class outranks every other.

    With gold class ``g``, ``p_wrong = max_{c != g} p_c`` and
    ``s = alpha / (1 - p_g + p_wrong)``::

        p~_c = s * p_c      for c != g
        p~_g = 1 - s * (1 - p_g)

    The construction guarantees ``sum(p~) == 1`` and
    ``p~_g - s * p_wrong == 1 - alpha > 0``, so for ``0 < alpha < 1`` the gold
    class is strictly top-1 afterwards and non-target *ratios* are preserved.

    This is an application of existing LoCa, not a new calibration method, and it
    consumes the ground-truth label — the run that uses it is label-exposed even
    without a CE term (`experiment_protocol.md` §8).

    Args:
        teacher_probs: ``[..., V]``, each row summing to 1.
        targets: ``[...]`` gold class indices. Must not contain ``IGNORE_INDEX``;
            select valid positions before calling.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    if (targets == IGNORE_INDEX).any():
        raise ValueError(
            "targets contain IGNORE_INDEX; select valid answer positions before calibrating")
    probs = teacher_probs.float()
    index = targets.unsqueeze(-1)

    p_gold = probs.gather(-1, index)                       # [..., 1]
    # Highest NON-target class. Masking gold out first matters: when gold is not
    # already top-1, the highest wrong class *is* top-1, and taking "second
    # highest" would select the wrong competitor.
    without_gold = probs.scatter(-1, index, float("-inf"))
    p_wrong = without_gold.max(dim=-1, keepdim=True).values

    scale = alpha / (1.0 - p_gold + p_wrong).clamp_min(1e-12)
    calibrated = probs * scale
    calibrated.scatter_(-1, index, 1.0 - scale.squeeze(-1).unsqueeze(-1) * (1.0 - p_gold))
    return calibrated


def loca_kd_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                 labels: torch.Tensor, temperature: float = 1.0, alpha: float = 0.8,
                 already_shifted: bool = False) -> torch.Tensor:
    """KL against the LoCa-calibrated teacher, over valid answer positions."""
    if not already_shifted:
        teacher_logits, _ = shift_for_causal_lm(teacher_logits, labels)
        student_logits, labels = shift_for_causal_lm(student_logits, labels)
    mask = valid_answer_mask(labels)
    if not mask.any():
        raise ValueError("no valid answer positions: every label is IGNORE_INDEX")

    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    calibrated = loca_calibrate(teacher_probs, _safe_targets(labels, mask), alpha=alpha)
    return _kl_over_valid(calibrated, student_log_probs, mask, temperature)


# ---------------------------------------------------------------------------
# Feature transfer
# ---------------------------------------------------------------------------

def feature_transfer_loss(student_features: torch.Tensor, teacher_features: torch.Tensor,
                          kind: str = "cosine") -> torch.Tensor:
    """Negative-free feature alignment: the simpler alternative to contrastive.

    Retained as a mandatory control — if this matches contrastive alignment, the
    plan's outcome rules require using it and dropping the contrastive claim.
    """
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"feature shapes differ: {tuple(student_features.shape)} vs "
            f"{tuple(teacher_features.shape)}")
    student, teacher = student_features.float(), teacher_features.float()
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(F.normalize(student, dim=-1), F.normalize(teacher, dim=-1))
    raise ValueError(f"unknown feature loss {kind!r}; expected 'cosine' or 'mse'")


def contrastive_loss(student_features: torch.Tensor, positive_features: torch.Tensor,
                     negative_bank: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """One-directional student-to-teacher contrastive loss against a memory bank.

    The legacy implementation built an ``[B, B]`` similarity matrix from the
    physical batch. At the pilot's batch size of one that is a ``[1, 1]`` matrix
    whose cross-entropy is *identically zero* — no negatives exist, so the term
    contributes nothing and no gradient flows. Gradient accumulation does not
    help: it enlarges the optimizer step, not the candidate set.

    Negatives therefore come from an explicit bank of cached teacher features
    drawn from **other training scenes**, and an empty bank is an error.

    Args:
        student_features: ``[N, D]``
        positive_features: ``[N, D]`` teacher feature of each student's own scene.
        negative_bank: ``[M, D]``, ``M >= 1``, excluding the scene and its
            room/sequence neighbours.
    """
    if negative_bank.dim() != 2 or negative_bank.size(0) < 1:
        raise ValueError(
            "contrastive loss requires a non-empty negative bank. A single-scene "
            "batch yields an identically zero loss and no gradient (audit B4).")
    if student_features.shape != positive_features.shape:
        raise ValueError("student and positive feature shapes differ")
    if negative_bank.size(-1) != student_features.size(-1):
        raise ValueError("negative bank dimension does not match the student features")

    student = F.normalize(student_features.float(), dim=-1)
    positive = F.normalize(positive_features.float(), dim=-1)
    negatives = F.normalize(negative_bank.float(), dim=-1)

    positive_logit = (student * positive).sum(-1, keepdim=True)       # [N, 1]
    negative_logits = student @ negatives.T                            # [N, M]
    logits = torch.cat([positive_logit, negative_logits], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long,
                                               device=logits.device))
