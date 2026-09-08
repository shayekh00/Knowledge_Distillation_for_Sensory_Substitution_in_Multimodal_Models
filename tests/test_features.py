"""The feature-alignment path is correct, or refuses to run.

Every failure mode covered here is silent. Splitting the packed vision sequence
on a wrong grid still produces correctly-shaped features — just features that
mix one image's patches into another's mean. A negative bank that contains a
room neighbour of its own positive still trains, just against a false negative.
Neither raises on its own, so each is pinned by a test that checks the value and
not merely the shape.
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
from distillation.features import (  # noqa: E402
    AlignmentHead,
    merged_token_counts,
    pool_per_image,
    sample_negative_bank,
)
from distillation.teacher_cache_loader import FeatureCache, QwenAdapter  # noqa: E402

MERGE = 2


def feature_key(**overrides):
    fields = {
        "dataset_version": "v2.4", "split": "train",
        "teacher_model": "Qwen/Qwen3.5-9B", "precision": "bfloat16",
        "signal_kind": "pooled_features",
        "feature_layer": "visual.pooler_output(post_merger)",
        "crop_aggregation": "mean over merged tokens, L2-normalised",
    }
    fields.update(overrides)
    return CacheKey(fields)


def build_feature_cache(tmp_path, image_ids, sequence_ids, dim=8, key=None):
    directory = str(tmp_path / "fcache")
    os.makedirs(directory, exist_ok=True)
    key = key or feature_key()
    write_cache_key(directory, key)
    rows = len(image_ids)
    features = np.arange(rows * dim, dtype=np.float32).reshape(rows, dim)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    np.savez_compressed(os.path.join(directory, "feature_table.npz"),
                        features=features.astype(np.float16),
                        image_ids=np.array(image_ids),
                        sequence_ids=np.array(sequence_ids))
    return directory, key


# ---------------------------------------------------------------- token counts

def test_merged_token_counts_divides_by_the_merge_block():
    grid = torch.tensor([[1, 34, 46], [1, 28, 36]])
    assert merged_token_counts(grid, MERGE).tolist() == [391, 252]


def test_merged_token_counts_rejects_a_grid_it_would_have_to_truncate():
    # 1*3*3 = 9 is not divisible by 4; floor division would drop two patches
    # from this image's pooled feature and raise nothing.
    with pytest.raises(ValueError, match="not divisible"):
        merged_token_counts(torch.tensor([[1, 3, 3]]), MERGE)


def test_merged_token_counts_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match=r"\[B, 3\]"):
        merged_token_counts(torch.tensor([1, 34, 46]), MERGE)


# ---------------------------------------------------------------- pooling

def test_pool_per_image_means_each_image_over_its_own_tokens_only():
    # Two images of 1 and 2 merged tokens: grids 1*2*2=4 -> 1 and 1*2*4=8 -> 2.
    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    merged = torch.tensor([[3.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    pooled = pool_per_image(merged, grid, MERGE, normalize=False)
    # image 0 = row 0 alone; image 1 = mean(rows 1,2) = 3.0 — deliberately the
    # same value as image 0, so a test that only checked shapes would pass even
    # if the split were wrong.
    assert pooled.tolist() == [[3.0, 0.0], [3.0, 0.0]]


def test_pool_per_image_does_not_bleed_one_image_into_another():
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    merged = torch.tensor([[1.0, 0.0], [9.0, 0.0]])
    pooled = pool_per_image(merged, grid, MERGE, normalize=False)
    assert pooled[0].tolist() == [1.0, 0.0]
    assert pooled[1].tolist() == [9.0, 0.0]


def test_pool_per_image_l2_normalises_by_default():
    grid = torch.tensor([[1, 2, 2]])
    pooled = pool_per_image(torch.tensor([[3.0, 4.0]]), grid, MERGE)
    assert pooled.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_pool_per_image_rejects_a_grid_that_does_not_account_for_the_tensor():
    grid = torch.tensor([[1, 2, 2]])                 # accounts for 1 token
    merged = torch.zeros(5, 2)                        # but 5 are present
    with pytest.raises(ValueError, match="merged tokens"):
        pool_per_image(merged, grid, MERGE)


def test_pool_per_image_is_differentiable_into_the_vision_tower():
    grid = torch.tensor([[1, 2, 4]])
    merged = torch.randn(2, 4, requires_grad=True)
    pool_per_image(merged, grid, MERGE).sum().backward()
    assert merged.grad is not None and merged.grad.abs().sum() > 0


def test_pool_per_image_accumulates_in_float32_from_bfloat16_input():
    grid = torch.tensor([[1, 2, 2]])
    pooled = pool_per_image(torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16), grid, MERGE)
    assert pooled.dtype == torch.float32


# ---------------------------------------------------------------- alignment head

def test_alignment_head_maps_student_width_to_teacher_width():
    head = AlignmentHead(1024, 4096)
    assert head(torch.randn(3, 1024)).shape == (3, 4096)


def test_alignment_head_output_is_unit_norm():
    head = AlignmentHead(6, 10)
    norms = head(torch.randn(4, 6)).norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_alignment_head_rejects_the_wrong_input_width():
    head = AlignmentHead(6, 10)
    with pytest.raises(ValueError, match="width 6"):
        head(torch.randn(2, 7))


def test_alignment_head_has_no_bias_so_a_gain_is_not_extra_capacity():
    assert AlignmentHead(4, 8).project.bias is None


# ---------------------------------------------------------------- negative bank

def test_negative_bank_excludes_the_batch_own_sequences():
    features = torch.eye(5)
    sequences = ["s0", "s0", "s1", "s2", "s3"]
    bank, scenes = sample_negative_bank(features, sequences, {"s0"}, size=10)
    # rows 0 and 1 are both sequence s0 and must not appear at all.
    assert scenes == 3
    assert not any(torch.equal(row, features[0]) or torch.equal(row, features[1])
                   for row in bank)


def test_negative_bank_draws_one_row_per_distinct_scene():
    features = torch.eye(6)
    sequences = ["a", "a", "a", "b", "b", "c"]   # 3 scenes, 6 rows
    bank, scenes = sample_negative_bank(features, sequences, set(), size=10)
    assert scenes == 3 and bank.size(0) == 3


def test_negative_bank_caps_at_the_requested_size():
    features = torch.eye(20)
    bank, scenes = sample_negative_bank(features, [f"s{i}" for i in range(20)],
                                        set(), size=7)
    assert bank.size(0) == 7 and scenes == 7


def test_negative_bank_refuses_when_every_scene_is_a_neighbour():
    # The audit's B4 pathology: a single-scene candidate set makes the
    # contrastive loss identically zero with no gradient, and never raises.
    features = torch.eye(3)
    with pytest.raises(ValueError, match="single scene"):
        sample_negative_bank(features, ["s0", "s0", "s0"], {"s0"}, size=5)


def test_negative_bank_is_reproducible_under_a_seeded_generator():
    features = torch.randn(30, 4)
    sequences = [f"s{i}" for i in range(30)]
    first = sample_negative_bank(features, sequences, set(), 5,
                                 torch.Generator().manual_seed(17))[0]
    second = sample_negative_bank(features, sequences, set(), 5,
                                  torch.Generator().manual_seed(17))[0]
    assert torch.equal(first, second)


def test_negative_bank_rejects_mismatched_id_count():
    with pytest.raises(ValueError, match="sequence ids"):
        sample_negative_bank(torch.eye(4), ["a", "b"], set(), 2)


# ---------------------------------------------------------------- FeatureCache

def test_feature_cache_refuses_a_key_it_was_not_built_for(tmp_path):
    directory, _ = build_feature_cache(tmp_path, ["i0", "i1"], ["s0", "s1"])
    with pytest.raises(Exception):
        FeatureCache(directory, feature_key(split="val"))


def test_feature_cache_returns_the_positive_of_each_row(tmp_path):
    directory, key = build_feature_cache(tmp_path, ["i0", "i1", "i2"],
                                         ["s0", "s1", "s2"], dim=8)
    cache = FeatureCache(directory, key)
    rows = [{"image_id": "i2", "sequence_id": "s2"},
            {"image_id": "i0", "sequence_id": "s0"}]
    signals = cache.signals_for(rows, bank_size=1)
    # Order follows the batch, not the table: row 0 of the batch is image i2.
    assert torch.allclose(signals.features[0], cache.features[2])
    assert torch.allclose(signals.features[1], cache.features[0])


def test_feature_cache_bank_excludes_every_sequence_in_the_batch(tmp_path):
    directory, key = build_feature_cache(
        tmp_path, [f"i{i}" for i in range(6)], ["s0", "s0", "s1", "s2", "s3", "s4"])
    cache = FeatureCache(directory, key)
    rows = [{"image_id": "i0", "sequence_id": "s0"},
            {"image_id": "i3", "sequence_id": "s2"}]
    signals = cache.signals_for(rows, bank_size=99)
    # s0 (rows 0,1) and s2 (row 3) are excluded; s1, s3, s4 remain.
    assert signals.metadata["distinct_negative_scenes"] == 3
    for excluded in (0, 1, 3):
        assert not any(torch.allclose(row, cache.features[excluded])
                       for row in signals.negative_bank)


def test_feature_cache_reports_the_scene_count_for_logging(tmp_path):
    directory, key = build_feature_cache(tmp_path, ["i0", "i1", "i2"],
                                         ["s0", "s1", "s2"])
    signals = FeatureCache(directory, key).signals_for(
        [{"image_id": "i0", "sequence_id": "s0"}], bank_size=99)
    assert signals.metadata["distinct_negative_scenes"] == 2
    assert signals.metadata["feature_layer"] == "visual.pooler_output(post_merger)"


def test_feature_cache_raises_on_an_image_it_does_not_hold(tmp_path):
    directory, key = build_feature_cache(tmp_path, ["i0"], ["s0"])
    with pytest.raises(KeyError, match="absent from the feature cache"):
        FeatureCache(directory, key).signals_for(
            [{"image_id": "nope", "sequence_id": "s9"}])


def test_feature_cache_rejects_a_table_with_duplicate_images(tmp_path):
    directory, key = build_feature_cache(tmp_path, ["i0", "i0"], ["s0", "s1"])
    with pytest.raises(ValueError, match="duplicate image_ids"):
        FeatureCache(directory, key)


def test_feature_cache_promotes_float16_storage_to_float32(tmp_path):
    directory, key = build_feature_cache(tmp_path, ["i0", "i1"], ["s0", "s1"])
    assert FeatureCache(directory, key).features.dtype == torch.float32


# ---------------------------------------------------------------- adapter guard

class _StubVisionModel:
    class config:
        class vision_config:
            spatial_merge_size = 2


def test_adapter_requires_an_alignment_head_for_a_feature_objective():
    adapter = QwenAdapter(_StubVisionModel(), spatial_merge_size=2)
    with pytest.raises(ValueError, match="AlignmentHead"):
        adapter.student_features({"pixel_values": torch.zeros(4, 3),
                                  "image_grid_thw": torch.tensor([[1, 2, 2]])})


def test_adapter_rejects_a_batch_with_no_image():
    adapter = QwenAdapter(_StubVisionModel(), alignment_head=AlignmentHead(4, 8),
                          spatial_merge_size=2)
    with pytest.raises(ValueError, match="no pixel_values"):
        adapter.student_features({"input_ids": torch.zeros(1, 3, dtype=torch.long)})


def test_adapter_refuses_to_guess_the_merge_size():
    class NoConfig:
        pass
    with pytest.raises(ValueError, match="spatial_merge_size"):
        QwenAdapter(NoConfig())


# ------------------------------------------------- the no-op guard in RecipeConfig

def test_a_feature_objective_without_a_vision_surface_cannot_run():
    from distillation.runner import RecipeConfig
    config = RecipeConfig(recipe="D3x", stage="S2", use_ce=True,
                          feature_objective="contrastive",
                          trainable_modules=("language_attention",))
    with pytest.raises(ValueError, match="no vision parameters"):
        config.assert_trainable_surface_can_learn()


def test_such_a_recipe_still_constructs_so_the_library_stays_inspectable():
    # The check is at run launch, not construction: recipe_library() must remain
    # importable while D3/D5/D8 await the §13.1 point 5 decision.
    from distillation.runner import RecipeConfig, recipe_library
    assert recipe_library()["D3"].feature_objective == "contrastive"
    RecipeConfig(recipe="D3y", feature_objective="cosine")


def test_a_feature_objective_with_a_vision_surface_can_run():
    from distillation.runner import RecipeConfig
    config = RecipeConfig(recipe="D0x", stage="F", use_ce=False,
                          feature_objective="contrastive",
                          trainable_modules=("vision_attention",))
    config.assert_trainable_surface_can_learn()


def test_the_check_does_not_fire_for_rows_without_a_feature_objective():
    from distillation.runner import RecipeConfig
    config = RecipeConfig(recipe="X2x", stage="S2", use_ce=True, kd_objective="xtoken",
                          top_k=4096, trainable_modules=("language_attention",))
    config.assert_trainable_surface_can_learn()


# ------------------------------------------------- two-stage derivation (§13.1 pt 5)

def test_stage_one_is_feature_alignment_only_on_a_vision_surface():
    from distillation.runner import recipe_library
    one = recipe_library()["D8"].stage_one()
    assert one.stage == "F"
    assert one.feature_objective == "cosine"      # the row's own objective, kept
    assert one.use_ce is False and one.kd_objective == "none" and one.use_loca is False
    # Both the vision attention and the merger (the model's own vision->language
    # projector) are trainable in stage one — author decision 2026-09-07, so the
    # alignment loss can reshape the vision->language interface itself, not just
    # what happens upstream of it.
    assert one.trainable_modules == ("vision_attention", "vision_merger")
    one.assert_trainable_surface_can_learn()


def test_stage_two_drops_the_feature_term_and_carries_the_parent():
    from distillation.runner import recipe_library
    two = recipe_library()["D8"].stage_two("runs/kd/D8/stage_F")
    assert two.stage == "S2"
    # Cleared deliberately: S2 freezes vision, so a feature term here would be
    # the no-op the surface check refuses. The alignment arrives as weights.
    assert two.feature_objective == "none" and two.lambda_feature == 0.0
    assert two.parent_checkpoint == "runs/kd/D8/stage_F"
    assert two.use_ce is True and two.kd_objective == "xtoken" and two.use_loca is True
    two.assert_trainable_surface_can_learn()


def test_the_two_stage_split_keeps_d5_and_d8_distinguishable():
    # Their only declared difference is the feature objective, which lives in
    # stage one — so if the split dropped it the two rows would collapse.
    from distillation.runner import recipe_library
    library = recipe_library()
    assert library["D5"].stage_one().feature_objective == "contrastive"
    assert library["D8"].stage_one().feature_objective == "cosine"


def test_stage_one_is_refused_for_a_row_that_has_none():
    from distillation.runner import recipe_library
    with pytest.raises(ValueError, match="no feature objective"):
        recipe_library()["X2"].stage_one()


def test_stage_two_requires_a_parent_checkpoint():
    from distillation.runner import recipe_library
    with pytest.raises(ValueError, match="parent checkpoint"):
        recipe_library()["D8"].stage_two("")


def test_is_two_stage_tracks_the_feature_objective():
    from distillation.runner import recipe_library
    library = recipe_library()
    assert library["D8"].is_two_stage() and library["D5"].is_two_stage()
    assert library["D3"].is_two_stage()
    assert not library["X2"].is_two_stage() and not library["B3"].is_two_stage()


def test_deriving_stages_does_not_mutate_the_library_row():
    from distillation.runner import recipe_library
    row = recipe_library()["D8"]
    row.stage_one(); row.stage_two("somewhere")
    assert row.stage == "S2" and row.feature_objective == "cosine"
    assert row.parent_checkpoint is None


# ------------------------------------------------- vision_merger surface (2026-09-07)

def test_vision_merger_regex_is_the_two_merger_linears_only():
    # Real collision risk, checked because it is real: every one of the 12
    # vision blocks has an MLP leaf-named `linear_fc1`/`linear_fc2`, identical
    # to the merger's own leaf names. A suffix-style target would grab all 12.
    import re
    regex = r".*visual\.merger\.(linear_fc1|linear_fc2)$"
    assert re.match(regex, "base_model.model.model.visual.merger.linear_fc1")
    assert re.match(regex, "base_model.model.model.visual.merger.linear_fc2")
    assert not re.match(regex, "base_model.model.model.visual.blocks.0.mlp.linear_fc1")
    assert not re.match(regex, "base_model.model.model.visual.blocks.11.mlp.linear_fc2")
