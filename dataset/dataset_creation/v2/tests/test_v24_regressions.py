import json
import random

import numpy as np

import build_index
import balance
import existence
import freeze_release
import identify_superlative
import question_only
import relative_depth


class RecordingDropLogger:
    def __init__(self):
        self.rows = []

    def log(self, image_id, reason_code, detail=""):
        self.rows.append((image_id, reason_code, detail))


def make_object(
    concept,
    *,
    display_name=None,
    area_frac=0.05,
    depth_median_m=2.0,
    object_index=0,
    category="furniture",
):
    return {
        "object_index": object_index,
        "raw_name": concept,
        "concept": concept,
        "display_name": display_name or concept,
        "category": category,
        "is_structural": False,
        "in_vocab": True,
        "is_valid_polygon": True,
        "area_frac": area_frac,
        "centroid_x": 5.0,
        "centroid_y": 5.0,
        "depth_median_m": depth_median_m,
        "depth_valid_frac": 1.0,
        "eligible": True,
        "reference_eligible": True,
    }


def test_polygon_geometry_is_clipped_to_the_visible_image_bounds():
    annotation = {
        "objects": [{"name": "chair"}],
        "frames": [{
            "polygon": [{
                "object": 0,
                "x": [-10.0, 10.0, 10.0, -10.0],
                "y": [0.0, 0.0, 10.0, 10.0],
            }]
        }],
    }
    config = {"depth": {"min_valid_fraction": 0.3}}

    [record] = build_index.build_polygon_records(
        annotation,
        image_width=10,
        image_height=10,
        depth_m=np.ones((10, 10), dtype=np.float32),
        config=config,
        image_id="scene/1",
        drop_rows=[],
    )

    assert record["area_frac"] == 1.0
    assert 0.0 <= record["centroid_x"] <= 10.0
    assert 0.0 <= record["centroid_y"] <= 10.0
    assert record["touches_border"] is True


def test_relative_depth_answer_is_an_exact_member_of_display_answer_space():
    trash_can = make_object(
        "trashcan", display_name="trash_can", depth_median_m=1.0, object_index=0
    )
    coffee_table = make_object(
        "coffeetable", display_name="coffee_table", depth_median_m=2.0, object_index=1
    )

    candidates = relative_depth.generate_candidates_for_scene(
        {"image_id": "scene/1"},
        [trash_can, coffee_table],
        random.Random(42),
        {},
        RecordingDropLogger(),
    )

    assert len(candidates) == 1
    [candidate] = candidates
    assert candidate["answer"] in candidate["answer_space"].split("|")
    assert "_" not in candidate["answer_space"]


def test_identify_superlative_generates_only_depth_grounded_variants():
    near_large = make_object(
        "chair", area_frac=0.30, depth_median_m=1.0, object_index=0
    )
    far_small = make_object(
        "table", area_frac=0.10, depth_median_m=2.0, object_index=1
    )

    candidates = identify_superlative.generate_candidates_for_scene(
        {"image_id": "scene/1"},
        [near_large, far_small],
        random.Random(42),
        {},
        RecordingDropLogger(),
    )

    assert {candidate["variant"] for candidate in candidates} == {
        "closest_camera",
        "farthest_camera",
    }


def test_existence_emits_every_plausible_absent_concept_for_global_balancing(monkeypatch):
    canonical_vocab = {
        "chair": {"display_name": "chair", "category": "furniture", "is_structural": False},
        "sofa": {"display_name": "sofa", "category": "furniture", "is_structural": False},
        "table": {"display_name": "table", "category": "furniture", "is_structural": False},
        "toilet": {"display_name": "toilet", "category": "fixture", "is_structural": False},
    }
    monkeypatch.setattr(existence, "_canonical_vocab_cache", canonical_vocab)
    monkeypatch.setattr(existence, "_scene_type_presence_fraction", {"office": {}})
    monkeypatch.setattr(
        existence,
        "_all_concepts_by_category",
        {"furniture": {"chair", "sofa", "table"}, "fixture": {"toilet"}},
    )
    chair = make_object("chair", object_index=0)

    candidates = existence.generate_candidates_for_scene(
        {"image_id": "scene/1", "scene_type": "office"},
        [chair],
        random.Random(42),
        {},
        RecordingDropLogger(),
    )

    negative_concepts = {
        json.loads(json.dumps(candidate["evidence"]))["concept"]
        for candidate in candidates
        if candidate["answer"] == "no"
    }
    assert negative_concepts == {"sofa", "table"}


def test_existence_emits_every_present_concept_for_global_balancing(monkeypatch):
    canonical_vocab = {
        "chair": {"display_name": "chair", "category": "furniture", "is_structural": False},
        "table": {"display_name": "table", "category": "furniture", "is_structural": False},
    }
    monkeypatch.setattr(existence, "_canonical_vocab_cache", canonical_vocab)
    monkeypatch.setattr(existence, "_scene_type_presence_fraction", {"office": {}})
    monkeypatch.setattr(
        existence,
        "_all_concepts_by_category",
        {"furniture": {"chair", "table"}},
    )

    candidates = existence.generate_candidates_for_scene(
        {"image_id": "scene/1", "scene_type": "office"},
        [make_object("chair", object_index=0), make_object("table", object_index=1)],
        random.Random(42),
        {},
        RecordingDropLogger(),
    )

    positive_concepts = {
        candidate["evidence"]["concept"]
        for candidate in candidates
        if candidate["answer"] == "yes"
    }
    assert positive_concepts == {"chair", "table"}


def test_existence_balancing_is_exact_per_concept_and_never_reuses_an_image():
    rows = [
        {"image_id": "i1", "answer": "yes", "concept": "chair"},
        {"image_id": "i2", "answer": "yes", "concept": "chair"},
        {"image_id": "i3", "answer": "no", "concept": "chair"},
        {"image_id": "i4", "answer": "no", "concept": "chair"},
        {"image_id": "i3", "answer": "yes", "concept": "table"},
        {"image_id": "i5", "answer": "yes", "concept": "table"},
        {"image_id": "i1", "answer": "no", "concept": "table"},
        {"image_id": "i6", "answer": "no", "concept": "table"},
    ]
    candidates = __import__("pandas").DataFrame(rows)

    balanced = balance.balance_binary_per_group_and_image(
        candidates,
        answer_column="answer",
        group_column="concept",
        image_column="image_id",
        rng=random.Random(42),
    )

    assert balanced["image_id"].is_unique
    for _, group in balanced.groupby("concept"):
        answer_counts = group["answer"].value_counts().to_dict()
        assert answer_counts["yes"] == answer_counts["no"]
    assert balanced["_balance_pair_id"].value_counts().eq(2).all()


def test_pair_subsampling_never_splits_an_existence_balance_pair():
    candidates = __import__("pandas").DataFrame({
        "image_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "_balance_pair_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
    })

    sampled = balance.stratified_pair_subsample(
        candidates,
        target_size=4,
        pair_column="_balance_pair_id",
        stratify_column="sensor",
        rng=random.Random(42),
    )

    assert len(sampled) == 4
    assert sampled["_balance_pair_id"].value_counts().eq(2).all()


def test_type_cap_preserves_existence_pairs():
    pandas = __import__("pandas")
    frames = {
        question_type: pandas.DataFrame({"image_id": ["shared"]})
        for question_type in (
            "existence",
            "identify_superlative",
            "relative_depth",
            "nearest_object",
            "left_right",
        )
    }

    trimmed = balance.cap_distinct_types_per_image(
        frames,
        max_types_per_image=4,
        rng=random.Random(42),
        protected_question_types={"existence"},
    )

    assert len(trimmed["existence"]) == 1
    assert sum(len(frame) for frame in trimmed.values()) == 4


def test_grouped_answer_cap_removes_anchor_groups_that_cannot_meet_the_cap():
    pandas = __import__("pandas")
    candidates = pandas.DataFrame({
        "image_id": [f"i{index}" for index in range(10)],
        "anchor": ["desk"] * 8 + ["bed"] * 2,
        "answer": ["chair"] * 4 + ["table"] * 2 + ["sofa"] * 2 + ["pillow"] * 2,
    })

    capped = balance.cap_answer_share_per_group(
        candidates,
        answer_column="answer",
        group_column="anchor",
        max_share=0.40,
        rng=random.Random(42),
    )

    assert set(capped["anchor"]) == {"desk"}
    assert capped["answer"].value_counts(normalize=True).max() <= 0.40


def test_relative_depth_question_only_target_is_answer_position():
    row = {
        "answer": "coffee table",
        "answer_space": "trash can|coffee table",
    }
    assert question_only.target_label("relative_depth", row) == "second"


def test_question_only_evaluation_detects_a_large_language_shortcut():
    pandas = __import__("pandas")
    train = pandas.DataFrame({
        "question": ["Is a chair visible?"] * 20 + ["Is a sofa visible?"] * 20,
        "answer": ["yes"] * 20 + ["no"] * 20,
        "answer_space": ["yes|no"] * 40,
    })
    validation = train.copy()

    result = question_only.evaluate_question_type(
        "existence", train, validation, random_state=42
    )

    assert result["accuracy"] == 1.0
    assert result["majority_baseline"] == 0.5
    assert result["excess_over_majority"] == 0.5
    assert result["passes"] is False


def test_question_only_majority_baseline_is_measured_on_the_evaluation_split():
    pandas = __import__("pandas")
    train = pandas.DataFrame({
        "question": ["Is an object visible?"] * 40,
        "answer": ["yes"] * 30 + ["no"] * 10,
        "answer_space": ["yes|no"] * 40,
    })
    validation = pandas.DataFrame({
        "question": ["Is an object visible?"] * 20,
        "answer": ["yes"] * 5 + ["no"] * 15,
        "answer_space": ["yes|no"] * 20,
    })

    result = question_only.evaluate_question_type(
        "existence", train, validation, random_state=42
    )

    assert result["majority_baseline"] == 0.75


def test_frozen_manifest_tracks_pipeline_source_and_question_templates():
    tracked_relative_paths = {
        __import__("os").path.relpath(path, freeze_release.REPO_ROOT)
        for path in freeze_release.TRACKED_INPUTS
    }

    assert "dataset/dataset_creation/v2/build_release.py" in tracked_relative_paths
    assert "dataset/dataset_creation/v2/existence.py" in tracked_relative_paths
    assert "data/templates/existence.txt" in tracked_relative_paths
    assert "data/templates/relative_depth.txt" in tracked_relative_paths
