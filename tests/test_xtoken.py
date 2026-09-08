"""X-Token cross-tokenizer distillation tests (plan §7.3, WP3b).

Uses synthetic tokenizers with deliberately mismatched boundaries, so the
alignment and projection are verified without downloading a model — the §19
Phase 4 requirement to check this "before touching a real model".
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

from distillation.losses import token_kd_loss  # noqa: E402
from distillation.xtoken import (  # noqa: E402
    Span,
    VocabularyMapping,
    aggregate_spans,
    align_spans,
    alignment_covers_everything,
    build_vocabulary_mapping,
    load_or_build_mapping,
    mapping_report,
    omitted_teacher_mass,
    project_student_probs,
    projected_kl_loss,
    build_student_to_teacher_lookup,
    calibrate_topk_with_loca,
)


class FakeTokenizer:
    """Minimal tokenizer over a fixed piece list. Longest-match segmentation."""

    def __init__(self, pieces, revision="v1"):
        self.pieces = list(pieces)
        self.revision = revision
        self._vocab = {piece: index for index, piece in enumerate(self.pieces)}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids):
        return "".join(self.pieces[int(i)] for i in ids)

    def encode(self, text):
        out, cursor = [], 0
        while cursor < len(text):
            for length in range(len(text) - cursor, 0, -1):
                candidate = text[cursor:cursor + length]
                if candidate in self._vocab:
                    out.append(self._vocab[candidate])
                    cursor += length
                    break
            else:
                raise ValueError(f"cannot tokenize {text[cursor:]!r} with {self.revision}")
        return out


# A teacher that keeps "201" whole, and a student that must spell it out. This is
# exactly the boundary disagreement the plan calls out.
TEACHER_PIECES = ["201", " cm", "yes", "no", " ", "left", "right"]
STUDENT_PIECES = ["2", "0", "1", " cm", "yes", "no", " ", "left", "right"]


@pytest.fixture
def teacher_tokenizer():
    return FakeTokenizer(TEACHER_PIECES, revision="teacher-r1")


@pytest.fixture
def student_tokenizer():
    return FakeTokenizer(STUDENT_PIECES, revision="student-r1")


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------

def test_alignment_handles_many_to_one_boundaries():
    """"201" as one teacher token against "2","0","1" in the student."""
    spans = align_spans(["201"], ["2", "0", "1"])
    assert len(spans) == 1
    assert spans[0].teacher == (0, 1) and spans[0].student == (0, 3)


def test_alignment_splits_where_boundaries_agree():
    spans = align_spans(["201", " cm"], ["2", "0", "1", " cm"])
    assert [(s.teacher, s.student) for s in spans] == [((0, 1), (0, 3)), ((1, 2), (3, 4))]


def test_alignment_is_one_to_one_when_tokenizers_agree():
    pieces = ["left", " ", "right"]
    spans = align_spans(pieces, list(pieces))
    assert [(s.teacher, s.student) for s in spans] == [((0, 1), (0, 1)),
                                                       ((1, 2), (1, 2)),
                                                       ((2, 3), (2, 3))]


def test_alignment_covers_every_position():
    """§7.3: no teacher or student answer position may be left unassigned."""
    teacher = ["201", " cm", "yes"]
    student = ["2", "0", "1", " cm", "yes"]
    spans = align_spans(teacher, student)
    assert alignment_covers_everything(spans, len(teacher), len(student))


def test_alignment_rejects_different_underlying_text():
    with pytest.raises(ValueError, match="different text"):
        align_spans(["yes"], ["no"])


def test_alignment_of_empty_answer_is_empty():
    assert align_spans([], []) == []


# ---------------------------------------------------------------------------
# Vocabulary mapping
# ---------------------------------------------------------------------------

def test_mapping_prefers_exact_surface_matches(student_tokenizer, teacher_tokenizer):
    mapping = build_vocabulary_mapping(student_tokenizer, teacher_tokenizer)
    pairs = dict(zip(mapping.student_ids.tolist(), mapping.teacher_ids.tolist()))
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    for surface in ("yes", "no", "left", "right", " cm"):
        assert pairs[student_vocab[surface]] == teacher_vocab[surface]


def test_mapping_falls_back_to_retokenisation(student_tokenizer, teacher_tokenizer):
    """"2" has no teacher token; it re-tokenizes and takes the first piece."""
    mapping = build_vocabulary_mapping(student_tokenizer, teacher_tokenizer)
    pairs = dict(zip(mapping.student_ids.tolist(), mapping.teacher_ids.tolist()))
    # The teacher cannot represent "2" alone, so those entries are unmapped
    # rather than silently pointed at an unrelated token.
    assert mapping.retokenized + mapping.unmapped >= 3
    assert mapping.exact_matches >= 5
    assert 0.0 < mapping.exact_match_fraction <= 1.0
    assert set(pairs.values()) <= set(teacher_tokenizer.get_vocab().values())


def test_mapping_is_sparse_never_a_dense_matrix(student_tokenizer, teacher_tokenizer):
    """§7.3: assert no dense V_student x V_teacher array is allocated."""
    mapping = build_vocabulary_mapping(student_tokenizer, teacher_tokenizer)
    assert mapping.student_ids.dim() == 1 and mapping.teacher_ids.dim() == 1
    assert mapping.student_ids.numel() <= mapping.student_vocab_size
    dense_size = mapping.student_vocab_size * mapping.teacher_vocab_size
    assert mapping.student_ids.numel() + mapping.teacher_ids.numel() < dense_size


def test_mapping_cache_is_keyed_by_both_revisions(tmp_path, student_tokenizer,
                                                  teacher_tokenizer):
    """§7.3: a revision change invalidates rather than silently reusing."""
    first = load_or_build_mapping(student_tokenizer, teacher_tokenizer, str(tmp_path),
                                  "student-r1", "teacher-r1")
    cached = load_or_build_mapping(student_tokenizer, teacher_tokenizer, str(tmp_path),
                                   "student-r1", "teacher-r1")
    assert first.content_hash() == cached.content_hash()
    files_before = set(os.listdir(tmp_path))
    load_or_build_mapping(student_tokenizer, teacher_tokenizer, str(tmp_path),
                          "student-r1", "teacher-r2")      # teacher revision bumped
    assert set(os.listdir(tmp_path)) - files_before, "new revision must not reuse the cache"


def test_mapping_report_carries_the_manifest_fields(student_tokenizer, teacher_tokenizer):
    report = mapping_report(build_vocabulary_mapping(
        student_tokenizer, teacher_tokenizer, "student-r1", "teacher-r1"))
    for key in ("mapping_hash", "student_revision", "teacher_revision", "coverage",
                "exact_match_fraction", "is_identity"):
        assert key in report


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def identity_mapping(vocab_size):
    ids = torch.arange(vocab_size)
    return VocabularyMapping(student_ids=ids, teacher_ids=ids.clone(),
                             student_vocab_size=vocab_size, teacher_vocab_size=vocab_size,
                             exact_matches=vocab_size, retokenized=0, unmapped=0)


def test_identity_mapping_is_detected():
    assert identity_mapping(8).is_identity()


def test_projection_sums_mass_of_students_sharing_a_teacher_target():
    # Students 0 and 1 both map to teacher 0; student 2 maps to teacher 1.
    mapping = VocabularyMapping(
        student_ids=torch.tensor([0, 1, 2]), teacher_ids=torch.tensor([0, 0, 1]),
        student_vocab_size=3, teacher_vocab_size=2,
        exact_matches=3, retokenized=0, unmapped=0)
    projected = project_student_probs(torch.tensor([[0.5, 0.2, 0.3]]), mapping)
    assert torch.allclose(projected, torch.tensor([[0.7, 0.3]]))


def test_projection_drops_mass_for_unmapped_tokens():
    mapping = VocabularyMapping(
        student_ids=torch.tensor([0]), teacher_ids=torch.tensor([0]),
        student_vocab_size=2, teacher_vocab_size=2,
        exact_matches=1, retokenized=0, unmapped=1)
    projected = project_student_probs(torch.tensor([[0.6, 0.4]]), mapping)
    assert projected.sum().item() == pytest.approx(0.6)


def test_projection_rejects_a_vocabulary_size_mismatch():
    with pytest.raises(ValueError, match="columns but the mapping"):
        project_student_probs(torch.rand(1, 5), identity_mapping(8))


# ---------------------------------------------------------------------------
# P-KL
# ---------------------------------------------------------------------------

def test_pkl_reduces_to_token_kl_under_an_identity_mapping():
    """§7.3, the load-bearing test: X-Token must generalize ordinary token KL.

    With an identity mapping and full support retained, P-KL and token KD are the
    same quantity, so a disagreement here is a projection bug.
    """
    torch.manual_seed(0)
    vocab = 12
    teacher_logits = torch.randn(1, 5, vocab)
    student_logits = torch.randn(1, 5, vocab)
    labels = torch.randint(0, vocab, (1, 5))

    token_kl = token_kd_loss(teacher_logits, student_logits, labels)

    # Same positions the shifted token loss uses.
    teacher_probs = F.softmax(teacher_logits[:, :-1, :].reshape(-1, vocab), dim=-1)
    student_probs = F.softmax(student_logits[:, :-1, :].reshape(-1, vocab), dim=-1)
    ids = torch.arange(vocab).expand(teacher_probs.size(0), -1)
    pkl = projected_kl_loss(ids, teacher_probs, student_probs, identity_mapping(vocab))

    assert pkl.item() == pytest.approx(token_kl.item(), abs=1e-5)


def test_pkl_is_near_zero_when_both_models_commit_to_the_same_text():
    """§7.3: identical underlying text under two different tokenizers."""
    # Teacher puts all mass on teacher token 0; the student splits its mass over
    # two student tokens that both map to teacher token 0.
    mapping = VocabularyMapping(
        student_ids=torch.tensor([0, 1]), teacher_ids=torch.tensor([0, 0]),
        student_vocab_size=3, teacher_vocab_size=2,
        exact_matches=2, retokenized=0, unmapped=0)
    teacher_ids = torch.tensor([[0, 1]])
    teacher_probs = torch.tensor([[1.0, 0.0]])
    student_probs = torch.tensor([[0.6, 0.4, 0.0]])
    assert projected_kl_loss(teacher_ids, teacher_probs, student_probs,
                             mapping).item() == pytest.approx(0.0, abs=1e-5)


def test_pkl_is_positive_when_the_models_disagree():
    mapping = identity_mapping(3)
    teacher_ids = torch.tensor([[0, 1, 2]])
    teacher_probs = torch.tensor([[0.9, 0.05, 0.05]])
    student_probs = torch.tensor([[0.05, 0.05, 0.9]])
    assert projected_kl_loss(teacher_ids, teacher_probs, student_probs,
                             mapping).item() > 0.5


def test_pkl_rejects_a_span_count_mismatch():
    with pytest.raises(ValueError, match="span count differs"):
        projected_kl_loss(torch.zeros(2, 3, dtype=torch.long), torch.rand(2, 3),
                          torch.rand(1, 4), identity_mapping(4))


def test_pkl_rejects_mismatched_topk_shapes():
    with pytest.raises(ValueError, match="same shape"):
        projected_kl_loss(torch.zeros(1, 3, dtype=torch.long), torch.rand(1, 4),
                          torch.rand(1, 4), identity_mapping(4))


# ---------------------------------------------------------------------------
# Top-K
# ---------------------------------------------------------------------------

def test_omitted_mass_is_reported_not_silently_dropped():
    """§7.3: reducing K must surface the omitted mass."""
    full = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    assert omitted_teacher_mass(full).item() == pytest.approx(0.0, abs=1e-6)
    truncated = full[:, :2]                      # keep top-2, drop 0.20
    assert omitted_teacher_mass(truncated).item() == pytest.approx(0.20, abs=1e-6)


def test_smaller_k_omits_more_mass():
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    masses = [omitted_teacher_mass(probs[:, :k]).item() for k in (4, 3, 2, 1)]
    assert masses == sorted(masses), "omitted mass must grow as K shrinks"


# ---------------------------------------------------------------------------
# Span aggregation
# ---------------------------------------------------------------------------

def test_span_aggregation_averages_positions_within_a_span():
    spans = [Span(teacher=(0, 1), student=(0, 3), text="201")]
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    aggregated = aggregate_spans(student, spans, which="student")
    assert aggregated.shape == (1, 2)
    assert torch.allclose(aggregated, torch.tensor([[0.5, 0.5]]))


def test_span_aggregation_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="teacher.*student"):
        aggregate_spans(torch.rand(2, 2), [Span((0, 1), (0, 1), "x")], which="both")


# ---------------------------------------------------------------------------
# Top-K-restricted LoCa (plan §8, D2/D5/D8) — compose_loss's xtoken branch
# silently ignored config.use_loca before this; these pin the fix.
# ---------------------------------------------------------------------------

def test_build_student_to_teacher_lookup_maps_known_ids_and_flags_the_rest():
    mapping = VocabularyMapping(
        student_ids=torch.tensor([0, 2, 5]), teacher_ids=torch.tensor([10, 12, 15]),
        student_vocab_size=6, teacher_vocab_size=20,
        exact_matches=3, retokenized=0, unmapped=3)
    lookup = build_student_to_teacher_lookup(mapping)
    assert lookup.tolist() == [10, -1, 12, -1, -1, 15]


def test_topk_loca_makes_gold_top1_when_found_in_cache():
    """Mirrors `test_loca_makes_gold_top1_whether_or_not_it_started_there`,
    through the top-K wrapper rather than `loca_calibrate` directly."""
    topk_ids = torch.tensor([[3, 1, 4]])
    topk_probs = torch.tensor([[0.70, 0.25, 0.05]])
    for gold_id in (3, 1, 4):
        calibrated, found = calibrate_topk_with_loca(
            topk_ids, topk_probs, torch.tensor([gold_id]), alpha=0.8)
        assert found.tolist() == [True]
        gold_col = (topk_ids[0] == gold_id).nonzero(as_tuple=True)[0].item()
        assert calibrated.argmax(-1).item() == gold_col, f"gold id {gold_id} not top-1"


def test_topk_loca_preserves_non_target_ratios():
    topk_ids = torch.tensor([[10, 11, 12, 13]])
    topk_probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    calibrated, found = calibrate_topk_with_loca(
        topk_ids, topk_probs, torch.tensor([10]), alpha=0.8)
    assert found.tolist() == [True]
    before = topk_probs[0, 1] / topk_probs[0, 2]
    after = calibrated[0, 1] / calibrated[0, 2]
    assert before.item() == pytest.approx(after.item(), abs=1e-5)


def test_topk_loca_gold_at_the_last_rank_of_the_cache():
    """Boundary case: gold is the K-th (last) retained entry, not dropped."""
    topk_ids = torch.tensor([[7, 8, 9, 42]])
    topk_probs = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
    calibrated, found = calibrate_topk_with_loca(
        topk_ids, topk_probs, torch.tensor([42]), alpha=0.8)
    assert found.tolist() == [True]
    assert calibrated.argmax(-1).item() == 3
    # p_wrong is the true global max here since top-K keeps the K largest —
    # 0.60, the highest of the three competitors, same as the dense case.
    expected_scale = 0.8 / (1 - 0.05 + 0.60)
    assert calibrated[0, 0].item() == pytest.approx(0.60 * expected_scale, abs=1e-5)


def test_topk_loca_excludes_rows_where_gold_is_missing_from_the_cache():
    """The critical new case top-K adds over the dense original: gold can
    legitimately fall outside a truncated cache. Silently approximating its
    probability would be exactly the failure mode already caught once this
    session (the enable_thinking prompt defect) — so these rows are dropped
    and counted, never guessed."""
    topk_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    topk_probs = torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    gold_teacher_ids = torch.tensor([2, 99])              # 99 is in neither row
    calibrated, found = calibrate_topk_with_loca(topk_ids, topk_probs, gold_teacher_ids, alpha=0.8)
    assert found.tolist() == [True, False]
    assert calibrated.shape == (1, 3)


def test_topk_loca_returns_empty_when_gold_is_found_nowhere():
    topk_ids = torch.tensor([[1, 2, 3]])
    topk_probs = torch.tensor([[0.5, 0.3, 0.2]])
    calibrated, found = calibrate_topk_with_loca(
        topk_ids, topk_probs, torch.tensor([999]), alpha=0.8)
    assert found.tolist() == [False]
    assert calibrated.shape == (0, 3)


def test_topk_loca_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same shape"):
        calibrate_topk_with_loca(torch.zeros(1, 3), torch.zeros(1, 4), torch.tensor([0]))
    with pytest.raises(ValueError, match="rows"):
        calibrate_topk_with_loca(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([0]))


def test_scatter_add_check_tolerance_scales_with_collapse_size():
    """`check_scatter_add_accumulates` (distillation/verify_xtoken_identity.py)
    once used a fixed 1e-5 tolerance, tuned on the Qwen-Qwen pair's 665-way
    collapse. Measured on the real Gemma-4-12B-it -> Qwen3.5-0.8B mapping
    (2026-09-07): a 5,880-way collapse lands at a 2.8e-5 float32 summation
    error — confirmed as accumulation noise, not a projection bug, by an
    independent float64 rerun and a plain sequential-sum reproduction — which
    the old fixed tolerance would have wrongly rejected. This pins the fix
    (`max(1e-5, n_collapsed * 1e-8)`) with a synthetic large collapse, without
    downloading either real tokenizer."""
    from distillation.verify_xtoken_identity import check_scatter_add_accumulates

    n = 6000
    # n student ids (0..n-1) all collapse onto teacher id 0; one more student
    # id (n) maps to a distinct teacher id so the mapping is not degenerate.
    student_ids = torch.arange(n + 1)
    teacher_ids = torch.cat([torch.zeros(n, dtype=torch.long), torch.tensor([1])])
    mapping = VocabularyMapping(
        student_ids=student_ids, teacher_ids=teacher_ids,
        student_vocab_size=n + 1, teacher_vocab_size=2,
        exact_matches=n + 1, retokenized=0, unmapped=0)
    check_scatter_add_accumulates(mapping)  # must not raise


def test_scatter_add_check_still_rejects_a_real_miss():
    """A genuine bug (only half the mass reaching the collapsed id) must still
    fail regardless of collapse size — the scaled tolerance must not become
    so loose it stops catching real defects."""
    from distillation.verify_xtoken_identity import check_scatter_add_accumulates
    from distillation import verify_xtoken_identity as module

    n = 6000
    student_ids = torch.arange(n + 1)
    teacher_ids = torch.cat([torch.zeros(n, dtype=torch.long), torch.tensor([1])])
    mapping = VocabularyMapping(
        student_ids=student_ids, teacher_ids=teacher_ids,
        student_vocab_size=n + 1, teacher_vocab_size=2,
        exact_matches=n + 1, retokenized=0, unmapped=0)

    original = module.project_student_probs

    def broken_projection(probs, mapping):
        return original(probs, mapping) * 0.5

    module.project_student_probs = broken_projection
    try:
        with pytest.raises(SystemExit, match="did not sum all contributions"):
            check_scatter_add_accumulates(mapping)
    finally:
        module.project_student_probs = original
