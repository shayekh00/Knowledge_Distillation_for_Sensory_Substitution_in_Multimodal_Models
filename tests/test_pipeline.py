"""Tests for the runner, negative sampler, cache/manifest, candidate scoring,
and the depth-only inference harness (plan §7.2, §7.4, §8.2, §8.3, §6.3, §15)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.cache import (  # noqa: E402
    CacheKey,
    RunManifest,
    make_run_id,
    verify_cache_key,
    write_cache_key,
)
from distillation.inference_isolation import (  # noqa: E402
    FileAccessTracer,
    IsolationReport,
    assert_predictions_invariant_to_rgb,
)
from distillation.negative_sampler import (  # noqa: E402
    SceneBank,
    assert_valid_candidate_set,
    bank_statistics,
    build_scene_bank,
    candidate_set_report,
    eligible_negative_indices,
    sample_negatives,
)
from distillation.runner import (  # noqa: E402
    RecipeConfig,
    TeacherSignals,
    compose_loss,
    recipe_library,
)
from evaluation.candidate_scoring import (  # noqa: E402
    LengthPolicy,
    TokenTrie,
    argmax_candidate,
    candidate_distribution,
    constrained_decode,
    score_candidates,
)

# ---------------------------------------------------------------------------
# Negative sampler (§8.3)
# ---------------------------------------------------------------------------


def rows_for(scene_sequence_pairs):
    return [{"image_id": scene, "sequence_id": sequence}
            for scene, sequence in scene_sequence_pairs]


def test_bank_deduplicates_scenes():
    """Several questions about one image must not become several negatives."""
    bank = build_scene_bank(rows_for([("s1", "r1"), ("s1", "r1"), ("s2", "r2")]))
    assert bank.scene_ids == ["s1", "s2"]


def test_room_neighbours_are_excluded():
    """§8.3: exclude the scene and known room/sequence neighbours."""
    bank = build_scene_bank(rows_for([("s1", "roomA"), ("s2", "roomA"), ("s3", "roomB")]))
    eligible = eligible_negative_indices(bank, "s1")
    # s2 shares roomA with s1, so only s3 is a legitimate negative.
    assert [bank.scene_ids[i] for i in eligible] == ["s3"]


def test_sampling_is_deterministic_and_seed_dependent():
    bank = build_scene_bank(rows_for([(f"s{i}", f"r{i}") for i in range(50)]))
    first = sample_negatives(bank, "s0", n_negatives=10, seed=17)
    assert np.array_equal(first, sample_negatives(bank, "s0", n_negatives=10, seed=17))
    assert not np.array_equal(first, sample_negatives(bank, "s0", n_negatives=10, seed=42))


def test_sampling_never_returns_the_anchor_or_its_room():
    bank = build_scene_bank(rows_for([("a1", "roomA"), ("a2", "roomA")]
                                     + [(f"s{i}", f"r{i}") for i in range(30)]))
    sampled = sample_negatives(bank, "a1", n_negatives=10, seed=3)
    scenes = [bank.scene_ids[i] for i in sampled]
    assert "a1" not in scenes and "a2" not in scenes


def test_single_scene_bank_is_rejected():
    """The audit B4 case: one scene cannot supply a contrastive negative."""
    bank = build_scene_bank(rows_for([("only", "room")]))
    with pytest.raises(ValueError, match="no eligible negatives"):
        sample_negatives(bank, "only", n_negatives=4)


def test_too_small_a_pool_is_an_error_unless_explicitly_allowed():
    bank = build_scene_bank(rows_for([(f"s{i}", f"r{i}") for i in range(5)]))
    with pytest.raises(ValueError, match="requested"):
        sample_negatives(bank, "s0", n_negatives=255)
    assert len(sample_negatives(bank, "s0", n_negatives=255, allow_fewer=True)) == 4


def test_candidate_set_report_counts_distinct_scenes():
    bank = build_scene_bank(rows_for([(f"s{i}", f"r{i}") for i in range(20)]))
    sampled = sample_negatives(bank, "s0", n_negatives=8, seed=1)
    report = candidate_set_report(bank, "s0", sampled)
    assert report["n_negatives"] == 8 and report["n_distinct_scenes"] == 8
    assert_valid_candidate_set(report)


def test_duplicate_negatives_are_rejected_by_the_validator():
    with pytest.raises(ValueError, match="distinct scenes"):
        assert_valid_candidate_set({"n_negatives": 4, "n_distinct_scenes": 2,
                                    "n_distinct_sequences": 2, "anchor_sequence": "r"})


def test_bank_without_sequence_ids_is_rejected():
    with pytest.raises(ValueError, match="no sequence id"):
        SceneBank(scene_ids=["s1"], sequence_of={})


def test_bank_statistics_report_group_sizes():
    stats = bank_statistics(build_scene_bank(
        rows_for([("s1", "r1"), ("s2", "r1"), ("s3", "r2")])))
    assert stats["n_scenes"] == 3 and stats["n_sequences"] == 2


# ---------------------------------------------------------------------------
# Cache keys and manifests (§8.2, §15)
# ---------------------------------------------------------------------------

def base_key(**overrides):
    fields = {"dataset_version": "v2.4", "split": "train", "teacher_model": "qwen-9b",
              "precision": "nf4", "signal_kind": "topk_logits", "top_k": 4096}
    fields.update(overrides)
    return CacheKey(fields)


def test_cache_key_changes_with_precision_and_topk():
    assert base_key().digest() != base_key(precision="bf16").digest()
    assert base_key().digest() != base_key(top_k=2048).digest()


def test_cache_key_changes_with_either_tokenizer_revision():
    assert base_key().digest() != base_key(teacher_tokenizer_revision="r2").digest()
    assert base_key().digest() != base_key(student_tokenizer_revision="r2").digest()


def test_topk_cache_must_declare_k():
    with pytest.raises(ValueError, match="top_k must be set"):
        CacheKey({"dataset_version": "v2.4", "split": "train", "teacher_model": "t",
                  "precision": "nf4", "signal_kind": "topk_logits"})


def test_unknown_cache_field_is_rejected():
    with pytest.raises(ValueError, match="unknown cache key field"):
        base_key(nonsense="x")


def test_reading_a_mismatched_cache_names_the_differing_fields(tmp_path):
    write_cache_key(str(tmp_path), base_key())
    verify_cache_key(str(tmp_path), base_key())            # matching: fine
    with pytest.raises(ValueError, match="precision"):
        verify_cache_key(str(tmp_path), base_key(precision="bf16"))


def test_unidentified_cache_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="unidentified cache"):
        verify_cache_key(str(tmp_path), base_key())


def test_run_id_encodes_config_so_drift_is_visible():
    first = make_run_id("20260907", "qw9b2qw08b", "D6", 17, {"lr": 1e-5})
    second = make_run_id("20260907", "qw9b2qw08b", "D6", 17, {"lr": 2e-5})
    assert first != second, "two configs must not share a run id"
    assert first.startswith("20260907-qw9b2qw08b-D6-s17-")


def test_manifest_requires_xtoken_provenance():
    manifest = RunManifest(
        run_id=make_run_id("20260907", "gemma2qw", "X2", 17, {}),
        recipe="X2", seed=17, dataset_version="v2.4", student_model="qwen-0.8b",
        student_revision="r1", trainable_modules=["language_attention"],
        distillation_mode="xtoken", teacher_model="gemma-4-12b")
    with pytest.raises(ValueError, match="teacher_tokenizer_revision"):
        manifest.validate()
    manifest.teacher_tokenizer_revision = "t-r1"
    manifest.student_tokenizer_revision = "s-r1"
    manifest.xtoken_mapping_hash = "abc123"
    manifest.top_k = 4096
    manifest.validate()


def test_manifest_rejects_a_malformed_run_id():
    manifest = RunManifest(run_id="not-a-run-id", recipe="B3", seed=17,
                           dataset_version="v2.4", student_model="m",
                           student_revision="r", trainable_modules=[])
    with pytest.raises(ValueError, match="does not match"):
        manifest.validate()


def test_manifest_round_trips_to_disk(tmp_path):
    manifest = RunManifest(
        run_id=make_run_id("20260907", "none2qw", "B3", 17, {}),
        recipe="B3", seed=17, dataset_version="v2.4", student_model="qwen-0.8b",
        student_revision="r1", trainable_modules=["language_attention"])
    path = manifest.write(str(tmp_path))
    assert json.load(open(path))["recipe"] == "B3"


# ---------------------------------------------------------------------------
# Candidate scoring and constrained decoding (§6.3)
# ---------------------------------------------------------------------------

def test_trie_reports_only_legal_continuations():
    trie = TokenTrie([[1, 2], [1, 3], [4]])
    assert trie.allowed_next([]) == {1, 4}
    assert trie.allowed_next([1]) == {2, 3}
    assert trie.allowed_next([4]) == set()
    assert trie.is_complete([1, 2]) and not trie.is_complete([1])


def test_constrained_decoding_cannot_produce_an_illegal_answer():
    trie = TokenTrie([[1, 2], [3]])

    def logits(prefix):
        return {1: 0.1, 2: 5.0, 3: 9.0, 99: 100.0}     # 99 is illegal but dominant

    assert constrained_decode(trie, logits) == [3]


def test_constrained_decoding_follows_the_trie_through_multiple_tokens():
    trie = TokenTrie([[1, 2], [1, 3]])

    def logits(prefix):
        return {1: 5.0, 2: 0.0, 3: 9.0}

    assert constrained_decode(trie, logits) == [1, 3]


def test_empty_candidate_sequence_is_rejected():
    with pytest.raises(ValueError, match="empty answer sequence"):
        TokenTrie([[]])


class StubScorer:
    def __init__(self, table):
        self.table = table

    def token_logprobs(self, prefix, continuation):
        return [self.table[token] for token in continuation]


def test_length_policy_changes_the_ranking_not_just_the_scale():
    """Why the convention has to be declared before results are seen."""
    scorer = StubScorer({1: -0.4, 2: -0.4, 3: -0.4, 4: -0.6})
    candidates = {"long": [1, 2, 3], "short": [4]}
    summed = score_candidates(scorer, [], candidates, LengthPolicy("sum"))
    averaged = score_candidates(scorer, [], candidates, LengthPolicy("mean"))
    # sum penalises length: -1.2 against -0.6, so the short answer wins.
    assert argmax_candidate(summed) == "short"
    # mean removes it entirely: -0.4 against -0.6, so the long answer wins.
    assert argmax_candidate(averaged) == "long"


def test_candidate_distribution_is_normalised_and_order_independent():
    scores = {"yes": -0.2, "no": -1.4}
    distribution = candidate_distribution(scores)
    assert sum(distribution.values()) == pytest.approx(1.0)
    reordered = candidate_distribution({"no": -1.4, "yes": -0.2})
    assert distribution["yes"] == pytest.approx(reordered["yes"])


def test_candidate_distribution_rejects_a_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be positive"):
        candidate_distribution({"a": -1.0, "b": -2.0}, temperature=0.0)


def test_argmax_breaks_ties_deterministically():
    assert argmax_candidate({"b": -1.0, "a": -1.0}) == "a"


# ---------------------------------------------------------------------------
# Depth-only inference (§7.4)
# ---------------------------------------------------------------------------

def test_rgb_path_mutation_must_not_change_predictions():
    rows = [{"image_path": "rgb/1.jpg", "depth_path": "d/1.png"} for _ in range(3)]

    def depth_only(batch):
        return [row["depth_path"] for row in batch]

    assert assert_predictions_invariant_to_rgb(depth_only, rows)["status"] == "passed"


def test_a_model_that_reads_rgb_is_caught():
    rows = [{"image_path": "rgb/1.jpg", "depth_path": "d/1.png"}]

    def peeks_at_rgb(batch):
        return [row["image_path"] for row in batch]

    with pytest.raises(AssertionError, match="not depth-only"):
        assert_predictions_invariant_to_rgb(peeks_at_rgb, rows)


def test_tracer_records_forbidden_reads(tmp_path):
    rgb = tmp_path / "rgb_frame.txt"
    rgb.write_text("x", encoding="utf-8")
    with FileAccessTracer() as tracer:
        open(str(rgb), encoding="utf-8").read()
    with pytest.raises(AssertionError, match="forbidden path"):
        tracer.assert_untouched(["rgb_"])


def test_tracer_is_clean_when_only_depth_is_read(tmp_path):
    depth = tmp_path / "depth_frame.txt"
    depth.write_text("x", encoding="utf-8")
    with FileAccessTracer() as tracer:
        open(str(depth), encoding="utf-8").read()
    tracer.assert_untouched(["rgb_", "teacher_cache"])


def test_isolation_report_tolerates_inconclusive_denial_but_not_rgb_leakage():
    passing = IsolationReport(path_denial={"status": "inconclusive"},
                              rgb_invariance={"status": "passed"},
                              tracer_clean=True, depth_provenance="depth_bfx")
    assert passing.passed()
    failing = IsolationReport(path_denial={"status": "passed"},
                              rgb_invariance={"status": "failed"},
                              tracer_clean=True, depth_provenance="depth_bfx")
    assert not failing.passed()


# ---------------------------------------------------------------------------
# Runner (§7.2)
# ---------------------------------------------------------------------------

class StubAdapter:
    def __init__(self, vocab=8, length=4, feature_dim=6):
        torch.manual_seed(0)
        self._logits = torch.randn(1, length, vocab, requires_grad=True)
        self._features = torch.randn(2, feature_dim)
        self._labels = torch.randint(0, vocab, (1, length))

    def student_logits(self, batch):
        return self._logits

    def student_features(self, batch):
        return self._features

    def labels(self, batch):
        return self._labels

    def trainable_parameters(self, config):
        return [self._logits]


def test_recipe_with_no_objective_is_rejected():
    with pytest.raises(ValueError, match="no objective at all"):
        RecipeConfig(recipe="broken", use_ce=False)


def test_loca_cannot_be_composed_with_candidate_kd():
    """§7.3: the interaction has not been derived, so it is refused."""
    with pytest.raises(ValueError, match="LoCa is defined over a single"):
        RecipeConfig(recipe="bad", kd_objective="candidate", use_loca=True)


def test_xtoken_recipe_must_declare_top_k():
    with pytest.raises(ValueError, match="must declare top_k"):
        RecipeConfig(recipe="X2", kd_objective="xtoken")


def test_stage_f_rejects_a_ce_term():
    with pytest.raises(ValueError, match="feature alignment only"):
        RecipeConfig(recipe="D0", stage="F", use_ce=True, feature_objective="cosine")


def test_label_exposure_accounts_for_loca_not_just_ce():
    """§9.4: removing CE does not make a run label-free if LoCa consumes gold."""
    assert not RecipeConfig(recipe="D9", use_ce=False,
                            kd_objective="xtoken", top_k=4096).label_exposed()
    assert RecipeConfig(recipe="D2", use_ce=False, kd_objective="xtoken",
                        use_loca=True, top_k=4096).label_exposed()


def test_ce_only_recipe_produces_a_ce_component():
    adapter = StubAdapter()
    total, components = compose_loss(RecipeConfig(recipe="B3"), adapter, {}, TeacherSignals())
    assert set(components) == {"ce", "total"}
    assert total.requires_grad


def test_missing_teacher_signal_is_an_error_not_a_silent_skip():
    adapter = StubAdapter()
    config = RecipeConfig(recipe="D1", kd_objective="xtoken", top_k=4096)
    with pytest.raises(ValueError, match="cache did not supply"):
        compose_loss(config, adapter, {}, TeacherSignals(), xtoken_mapping=object())


def test_each_component_is_logged_separately():
    """A per-component log is what would have surfaced the under-scaled KD term."""
    adapter = StubAdapter()
    teacher = TeacherSignals(logits=torch.randn(1, 4, 8),
                             features=adapter.student_features({}).clone())
    config = RecipeConfig(recipe="D4", kd_objective="token", feature_objective="cosine")
    _total, components = compose_loss(config, adapter, {}, teacher)
    assert {"ce", "kd", "feature", "total"} <= set(components)


def test_contrastive_recipe_requires_a_negative_bank():
    adapter = StubAdapter()
    teacher = TeacherSignals(features=adapter.student_features({}).clone())
    config = RecipeConfig(recipe="D3", feature_objective="contrastive")
    with pytest.raises(ValueError, match="negative_bank"):
        compose_loss(config, adapter, {}, teacher)


def test_recipe_library_covers_the_matrix_rows():
    library = recipe_library()
    for row in ("B3", "B4", "D0", "D1", "D5", "X0", "X1", "X2", "X3", "X5"):
        assert row in library, f"matrix row {row} missing from the library"
    assert library["X3"].use_ce is False
    assert library["X2"].kd_objective == "xtoken"
