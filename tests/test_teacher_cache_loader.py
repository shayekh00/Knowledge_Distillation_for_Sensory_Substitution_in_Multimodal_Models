"""The teacher cache is read correctly, or refuses to be read at all.

Every failure mode here is silent by nature: a misread cache yields finite
losses and a training curve that looks fine, computed against the wrong targets.
`projected_kl_loss` only checks row *counts*, so counts alone are not evidence.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.cache import CacheKey, write_cache_key  # noqa: E402
from distillation.losses import IGNORE_INDEX  # noqa: E402
from distillation.runner import TeacherSignals  # noqa: E402
from distillation.teacher_cache_loader import (  # noqa: E402
    TeacherCache,
    assert_rows_align,
)

TOP_K = 8


def make_key(**overrides):
    fields = {
        "dataset_version": "v2.4", "split": "train",
        "teacher_model": "Qwen/Qwen3.5-9B", "precision": "bfloat16",
        "prompt_hash": "f6cc1b602803c6f1", "signal_kind": "topk_logits",
        "top_k": TOP_K, "temperature": 1.0,
    }
    fields.update(overrides)
    return CacheKey(fields)


def build_cache(tmp_path, rows_per_example, key=None):
    """A cache directory holding `rows_per_example[i]` rows for example i."""
    directory = str(tmp_path / "cache")
    os.makedirs(directory, exist_ok=True)
    write_cache_key(directory, key or make_key())
    for index, count in enumerate(rows_per_example):
        np.savez_compressed(
            os.path.join(directory, f"q{index}.npz"),
            question_id=f"q{index}",
            topk_ids=np.arange(count * TOP_K, dtype=np.int32).reshape(count, TOP_K),
            topk_probs=np.full((count, TOP_K), 1.0 / TOP_K, dtype=np.float16))
    return directory


def labels_with(rows_per_example, length=6):
    """Labels whose supervised positions, after the causal shift, number
    `rows_per_example[i]` for row i."""
    labels = torch.full((len(rows_per_example), length), IGNORE_INDEX)
    for index, count in enumerate(rows_per_example):
        # shift drops position 0, so supervision starts at index 1.
        labels[index, 1:1 + count] = 7
    return labels


def test_cache_built_for_another_configuration_is_refused(tmp_path):
    """The whole point of the key: a cache from a different run must not load."""
    directory = build_cache(tmp_path, [2], key=make_key(precision="nf4"))
    with pytest.raises(ValueError, match="precision"):
        TeacherCache(directory, make_key(precision="bfloat16"))


def test_unidentified_cache_is_refused(tmp_path):
    directory = str(tmp_path / "bare")
    os.makedirs(directory)
    with pytest.raises(FileNotFoundError, match="no cache_key.json"):
        TeacherCache(directory, make_key())


def test_rows_are_concatenated_in_batch_order(tmp_path):
    """Row *i* of the teacher must be the student's *i*th supervised position.

    `projected_kl_loss` pairs them positionally and cannot detect a permutation,
    so order is part of the contract rather than an implementation detail.
    """
    cache = TeacherCache(build_cache(tmp_path, [2, 1, 3]), make_key())
    signals = cache.signals_for(["q0", "q1", "q2"])

    assert signals.topk_ids.shape == (6, TOP_K)
    assert signals.metadata["rows_per_example"] == [2, 1, 3]
    # q1's single row is the fixture's row 0, and lands after q0's two rows.
    assert torch.equal(signals.topk_ids[2], torch.arange(TOP_K))
    # Requesting a different order returns a different concatenation.
    reordered = cache.signals_for(["q1", "q0", "q2"])
    assert reordered.metadata["rows_per_example"] == [1, 2, 3]
    assert not torch.equal(reordered.topk_ids, signals.topk_ids)


def test_missing_example_is_refused_rather_than_skipped(tmp_path):
    """A cache built over a smaller row set must not train on the subset."""
    cache = TeacherCache(build_cache(tmp_path, [2]), make_key())
    with pytest.raises(KeyError, match="absent from"):
        cache.signals_for(["q0", "q_missing"])


def test_row_counts_matching_the_batch_pass(tmp_path):
    cache = TeacherCache(build_cache(tmp_path, [2, 1, 3]), make_key())
    signals = cache.signals_for(["q0", "q1", "q2"])
    assert_rows_align(signals, labels_with([2, 1, 3]))


def test_per_example_misalignment_is_caught_even_when_totals_match(tmp_path):
    """The failure `projected_kl_loss` cannot see.

    Totals agree (6 == 6) so the loss runs happily, but every row after the
    first example is compared against the wrong position. Checking per-example
    counts is what turns this from a silent wrong-target run into a crash.
    """
    cache = TeacherCache(build_cache(tmp_path, [2, 1, 3]), make_key())
    signals = cache.signals_for(["q0", "q1", "q2"])

    assert sum(signals.metadata["rows_per_example"]) == 6
    with pytest.raises(ValueError, match="do not match the batch's supervised positions"):
        assert_rows_align(signals, labels_with([1, 2, 3]))


def test_compose_loss_xtoken_consumes_the_cache_shapes(tmp_path):
    """End to end on synthetic tensors: the cache's row count is what the
    X-Token branch produces from the student side.

    This is the mismatch that would otherwise surface only on a real GPU run —
    `compose_loss` previously softmaxed every [B, L] position and handed
    `projected_kl_loss` far more rows than any compact cache holds.
    """
    from distillation.runner import RecipeConfig, compose_loss
    from distillation.teacher_cache_loader import QwenAdapter  # noqa: F401
    from distillation.xtoken import VocabularyMapping

    rows_per_example = [2, 1, 3]
    cache = TeacherCache(build_cache(tmp_path, rows_per_example), make_key())
    signals = cache.signals_for(["q0", "q1", "q2"])
    labels = labels_with(rows_per_example)

    student_vocab = 11
    logits = torch.randn(len(rows_per_example), labels.size(1), student_vocab,
                         requires_grad=True)

    class StubAdapter:
        def student_logits(self, batch):
            return logits

        def labels(self, batch):
            return labels

        def student_features(self, batch):
            raise NotImplementedError

        def trainable_parameters(self, config):
            return [logits]

    # Identity-ish mapping: every student token maps to a teacher id inside the
    # fixture's id range, so the gather in projected_kl_loss is well defined.
    mapping = VocabularyMapping(
        student_ids=torch.arange(student_vocab),
        teacher_ids=torch.arange(student_vocab),
        student_vocab_size=student_vocab,
        teacher_vocab_size=int(signals.topk_ids.max()) + 1,
        exact_matches=student_vocab, retokenized=0, unmapped=0)

    config = RecipeConfig(recipe="X2", use_ce=False, kd_objective="xtoken", top_k=TOP_K)
    total, components = compose_loss(config, StubAdapter(), {}, signals,
                                     xtoken_mapping=mapping)

    assert torch.isfinite(total)
    assert "kd" in components
    total.backward()
    assert logits.grad is not None


def test_kd_term_actually_depends_on_the_teacher(tmp_path):
    """The KD loss must move when the teacher's distribution moves.

    `NEW_SUBMISSION.md` §4.1 records a previous submission that shipped a KD
    term which was effectively off. A plausible-looking finite loss is not
    evidence the signal is connected: a term that ignored its teacher entirely
    would still print a number that varies with the batch. So this compares two
    different teachers over identical student logits and requires the loss to
    differ.
    """
    from distillation.runner import RecipeConfig, compose_loss
    from distillation.xtoken import VocabularyMapping

    rows_per_example = [2, 1]
    cache = TeacherCache(build_cache(tmp_path, rows_per_example), make_key())
    labels = labels_with(rows_per_example)
    student_vocab = 11
    torch.manual_seed(0)
    logits = torch.randn(len(rows_per_example), labels.size(1), student_vocab)

    class StubAdapter:
        def student_logits(self, batch): return logits
        def labels(self, batch): return labels
        def student_features(self, batch): raise NotImplementedError
        def trainable_parameters(self, config): return [logits]

    signals = cache.signals_for(["q0", "q1"])
    mapping = VocabularyMapping(
        student_ids=torch.arange(student_vocab), teacher_ids=torch.arange(student_vocab),
        student_vocab_size=student_vocab,
        teacher_vocab_size=int(signals.topk_ids.max()) + 1,
        exact_matches=student_vocab, retokenized=0, unmapped=0)
    config = RecipeConfig(recipe="X2", use_ce=False, kd_objective="xtoken", top_k=TOP_K)

    _, flat = compose_loss(config, StubAdapter(), {}, signals, xtoken_mapping=mapping)

    # Same shapes and ids, but mass concentrated on one token instead of uniform.
    peaked = TeacherSignals(
        topk_ids=signals.topk_ids.clone(),
        topk_probs=torch.zeros_like(signals.topk_probs),
        metadata=dict(signals.metadata))
    peaked.topk_probs[:, 0] = 1.0
    _, sharp = compose_loss(config, StubAdapter(), {}, peaked, xtoken_mapping=mapping)

    assert flat["kd"] != pytest.approx(sharp["kd"], rel=1e-3), (
        "KD loss is identical under a uniform and a one-hot teacher — the term is "
        "not reading its teacher signal")


# ---------------------------------------------------------------------------
# use_loca on an xtoken recipe (D2/D5/D8) — compose_loss's xtoken branch
# referenced config.use_loca nowhere before this, so every declared
# xtoken+LoCa recipe silently ran plain raw KD instead. Found by reading the
# code, not by running it — the same class of defect NEW_SUBMISSION.md §4.1
# already records happening once.
# ---------------------------------------------------------------------------

def test_use_loca_is_not_silently_ignored_on_an_xtoken_recipe():
    from distillation.runner import RecipeConfig, compose_loss
    from distillation.xtoken import VocabularyMapping, build_student_to_teacher_lookup

    student_vocab = 6
    labels = labels_with([2])
    # Two different gold ids, deliberately not the teacher's rank-1 choice at
    # either position — otherwise LoCa's rescaling could coincidentally match
    # the raw distribution and the test would not be exercising anything.
    labels[0, 1] = 0
    labels[0, 2] = 1
    torch.manual_seed(0)
    logits = torch.randn(1, labels.size(1), student_vocab)

    class StubAdapter:
        def student_logits(self, batch): return logits
        def labels(self, batch): return labels
        def student_features(self, batch): raise NotImplementedError
        def trainable_parameters(self, config): return [logits]

    topk_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    topk_probs = torch.tensor([[0.10, 0.50, 0.10, 0.10, 0.10, 0.10],
                               [0.50, 0.10, 0.10, 0.10, 0.10, 0.10]])
    signals = TeacherSignals(topk_ids=topk_ids, topk_probs=topk_probs,
                             metadata={"rows_per_example": [2]})
    mapping = VocabularyMapping(
        student_ids=torch.arange(student_vocab), teacher_ids=torch.arange(student_vocab),
        student_vocab_size=student_vocab, teacher_vocab_size=student_vocab,
        exact_matches=student_vocab, retokenized=0, unmapped=0)
    lookup = build_student_to_teacher_lookup(mapping)

    config_raw = RecipeConfig(recipe="D2", use_ce=False, kd_objective="xtoken",
                              top_k=6, use_loca=False)
    config_loca = RecipeConfig(recipe="D2", use_ce=False, kd_objective="xtoken",
                               top_k=6, use_loca=True)

    _, raw = compose_loss(config_raw, StubAdapter(), {}, signals, xtoken_mapping=mapping)
    _, loca = compose_loss(config_loca, StubAdapter(), {}, signals,
                           xtoken_mapping=mapping, gold_teacher_lookup=lookup)

    assert raw["kd"] != pytest.approx(loca["kd"], rel=1e-3), (
        "use_loca=True produced the same loss as use_loca=False — LoCa is "
        "being silently ignored again")
    assert loca["loca_gold_found_rate"] == pytest.approx(1.0)


def test_use_loca_without_a_lookup_is_refused_not_silently_skipped(tmp_path):
    from distillation.runner import RecipeConfig, compose_loss
    from distillation.xtoken import VocabularyMapping

    labels = labels_with([2])
    torch.manual_seed(0)
    logits = torch.randn(1, labels.size(1), 6)

    class StubAdapter:
        def student_logits(self, batch): return logits
        def labels(self, batch): return labels
        def student_features(self, batch): raise NotImplementedError
        def trainable_parameters(self, config): return [logits]

    signals = TeacherSignals(
        topk_ids=torch.zeros(2, 6, dtype=torch.long), topk_probs=torch.ones(2, 6) / 6,
        metadata={"rows_per_example": [2]})
    mapping = VocabularyMapping(
        student_ids=torch.arange(6), teacher_ids=torch.arange(6),
        student_vocab_size=6, teacher_vocab_size=6,
        exact_matches=6, retokenized=0, unmapped=0)
    config = RecipeConfig(recipe="D2", use_ce=False, kd_objective="xtoken",
                          top_k=6, use_loca=True)

    with pytest.raises(ValueError, match="gold_teacher_lookup"):
        compose_loss(config, StubAdapter(), {}, signals, xtoken_mapping=mapping)


def test_use_loca_refuses_an_empty_calibrated_set():
    """Every gold id falls outside the cache — must fail loudly, not train on
    zero examples silently."""
    from distillation.runner import RecipeConfig, compose_loss
    from distillation.xtoken import VocabularyMapping, build_student_to_teacher_lookup

    labels = labels_with([2])
    labels[0, 1] = 0            # gold student id 0 for both supervised positions
    labels[0, 2] = 0
    torch.manual_seed(0)
    logits = torch.randn(1, labels.size(1), 6)

    class StubAdapter:
        def student_logits(self, batch): return logits
        def labels(self, batch): return labels
        def student_features(self, batch): raise NotImplementedError
        def trainable_parameters(self, config): return [logits]

    # Cache only ever holds teacher ids 0..5, but gold (student id 0) maps to
    # teacher id 99 — never present in any cached row.
    topk_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    signals = TeacherSignals(topk_ids=topk_ids, topk_probs=torch.ones(2, 6) / 6,
                             metadata={"rows_per_example": [2]})
    mapping = VocabularyMapping(
        student_ids=torch.tensor([0]), teacher_ids=torch.tensor([99]),
        student_vocab_size=6, teacher_vocab_size=100,
        exact_matches=1, retokenized=0, unmapped=5)
    lookup = build_student_to_teacher_lookup(mapping)
    config = RecipeConfig(recipe="D2", use_ce=False, kd_objective="xtoken",
                          top_k=6, use_loca=True)

    with pytest.raises(ValueError, match="found gold.*for 0 of"):
        compose_loss(config, StubAdapter(), {}, signals, xtoken_mapping=mapping,
                    gold_teacher_lookup=lookup)
