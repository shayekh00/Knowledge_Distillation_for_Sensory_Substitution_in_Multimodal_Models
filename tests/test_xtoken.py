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
