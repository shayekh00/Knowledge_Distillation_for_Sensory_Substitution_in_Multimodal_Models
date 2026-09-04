"""
Unit tests on synthetic scenes with known answers (plan §12, P2
acceptance: "All gates covered by tests"). Every gate described in
docs/DATASET_CREATION_PLAN.md §4 gets at least one passing and one
failing synthetic case here — real SUNRGBD data is not touched.
"""
import math

import numpy as np

import count
import existence
import identify_superlative
import left_right
import relative_depth
import scene_objects
import vocab
from depth_utils import backproject_to_camera_frame


def make_object(concept, area_frac=0.05, centroid_x=100.0, centroid_y=100.0,
                 depth_median_m=2.0, object_index=0, eligible=True,
                 is_valid_polygon=True, in_vocab=True):
    return {
        "object_index": object_index, "raw_name": concept, "concept": concept,
        "display_name": concept, "category": "furniture", "is_structural": False,
        "in_vocab": in_vocab, "is_valid_polygon": is_valid_polygon, "area_frac": area_frac,
        "centroid_x": centroid_x, "centroid_y": centroid_y,
        "depth_median_m": depth_median_m, "depth_valid_frac": 1.0,
        "eligible": eligible, "reference_eligible": eligible,
    }


# ---------------------------------------------------------------- vocab.py

def test_normalize_raw_name_fixes_the_double_s_inflect_bug():
    assert vocab.normalize_raw_name("glass") == "glass"
    assert vocab.normalize_raw_name("glasses") == "glass"
    assert vocab.normalize_raw_name("mattress") == "mattress"
    assert vocab.normalize_raw_name("mattresses") == "mattress"


def test_normalize_raw_name_keeps_plural_only_nouns():
    assert vocab.normalize_raw_name("clothes") == "clothes"


def test_normalize_raw_name_strips_trailing_digits_and_singularizes():
    assert vocab.normalize_raw_name("wall23") == "wall"
    assert vocab.normalize_raw_name("Chairs") == "chair"


# --------------------------------------------------- identify_superlative.py

def test_largest_candidate_passes_when_margin_clears():
    winner_obj = make_object("bed", area_frac=0.30)
    loser_obj = make_object("chair", area_frac=0.15)  # exactly 2x, clears 1.3x
    winner, runner_up = identify_superlative._largest_candidate([winner_obj, loser_obj])
    assert winner is winner_obj
    assert runner_up is loser_obj


def test_largest_candidate_fails_when_too_close():
    obj_a = make_object("bed", area_frac=0.20)
    obj_b = make_object("chair", area_frac=0.18)  # ratio 1.11 < 1.3
    winner, _ = identify_superlative._largest_candidate([obj_a, obj_b])
    assert winner is None


def test_largest_candidate_single_object_always_wins():
    only_obj = make_object("bed", area_frac=0.05)
    winner, runner_up = identify_superlative._largest_candidate([only_obj])
    assert winner is only_obj
    assert runner_up is None


def test_closest_candidate_rejects_below_minimum_depth():
    too_close = make_object("chair", depth_median_m=0.2)
    far_enough = make_object("desk", depth_median_m=2.0)
    winner, _ = identify_superlative._closest_candidate([too_close, far_enough])
    assert winner is None


def test_closest_candidate_passes_with_margin_and_valid_depth():
    near = make_object("chair", depth_median_m=1.0)
    far = make_object("desk", depth_median_m=1.3)  # 1.3x clears 1.2x margin
    winner, runner_up = identify_superlative._closest_candidate([near, far])
    assert winner is near
    assert runner_up is far


def test_farthest_candidate_passes_with_margin():
    near = make_object("chair", depth_median_m=1.0)
    far = make_object("desk", depth_median_m=1.3)
    winner, runner_up = identify_superlative._farthest_candidate([near, far])
    assert winner is far
    assert runner_up is near


def test_farthest_candidate_fails_when_too_close_together():
    obj_a = make_object("chair", depth_median_m=2.0)
    obj_b = make_object("desk", depth_median_m=2.1)  # ratio 1.05 < 1.2
    winner, _ = identify_superlative._farthest_candidate([obj_a, obj_b])
    assert winner is None


# ------------------------------------------------------- relative_depth.py

def test_clears_gap_uses_the_larger_of_absolute_and_relative_threshold():
    assert relative_depth._clears_gap(1.0, 1.35) is True   # gap 0.35 >= max(0.3, 0.15)
    assert relative_depth._clears_gap(1.0, 1.20) is False  # gap 0.20 < 0.3
    assert relative_depth._clears_gap(5.0, 5.80) is True   # gap 0.80 >= max(0.3, 0.75)
    assert relative_depth._clears_gap(5.0, 5.70) is False  # gap 0.70 < 0.75


# --------------------------------------------------------------- count.py

def test_plural_form_handles_irregular_and_compound_nouns():
    assert count._plural_form("chair") == "chairs"
    assert count._plural_form("shelf") == "shelves"
    assert count._plural_form("trash_can") == "trash cans"


# ------------------------------------------------------------ existence.py

def test_pick_hard_negative_excludes_present_and_requires_plausibility(monkeypatch):
    canonical_vocab = {
        "chair": {"category": "furniture", "is_structural": False},
        "sofa": {"category": "furniture", "is_structural": False},
        "toilet": {"category": "fixture", "is_structural": False},
        "wall": {"category": "structure", "is_structural": True},
    }
    monkeypatch.setattr(existence, "_scene_type_presence_fraction", {"office": {"chair": 0.9}})
    monkeypatch.setattr(existence, "_all_concepts_by_category", {"furniture": {"chair", "sofa"}})

    rng = _DeterministicChoiceRng()
    negative = existence._pick_hard_negative(
        rng, scene_type="office", present_concepts={"chair"},
        present_categories={"furniture"}, canonical_vocab=canonical_vocab,
    )
    # "sofa" shares the present furniture category and is absent -> plausible.
    # "toilet" is absent but implausible here (no category/co-occurrence link) -> excluded.
    # "chair" is present -> excluded. "wall" is structural -> excluded.
    assert negative == "sofa"


class _DeterministicChoiceRng:
    def choice(self, seq):
        return seq[0]


# ------------------------------------------------------------- left_right.py

def test_iou_of_non_overlapping_squares_is_zero():
    from shapely.geometry import Polygon
    square_a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    square_b = Polygon([(100, 0), (110, 0), (110, 10), (100, 10)])
    assert left_right._iou(square_a, square_b) == 0.0


def test_iou_of_identical_squares_is_one():
    from shapely.geometry import Polygon
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert left_right._iou(square, square) == 1.0


def test_passes_pair_gates_rejects_small_horizontal_gap():
    obj_a = make_object("chair", centroid_x=100.0)
    obj_b = make_object("desk", centroid_x=105.0)  # gap 5px on a 1000px-wide image
    assert left_right.passes_pair_gates(obj_a, obj_b, image_width=1000.0, iou=0.0) is False


def test_passes_pair_gates_rejects_high_overlap():
    obj_a = make_object("chair", centroid_x=100.0)
    obj_b = make_object("desk", centroid_x=900.0)
    assert left_right.passes_pair_gates(obj_a, obj_b, image_width=1000.0, iou=0.5) is False


def test_passes_pair_gates_accepts_clear_separated_pair():
    obj_a = make_object("chair", centroid_x=100.0)
    obj_b = make_object("desk", centroid_x=900.0)
    assert left_right.passes_pair_gates(obj_a, obj_b, image_width=1000.0, iou=0.0) is True


# ----------------------------------------------------------- scene_objects.py

def test_true_instance_counts_ignores_the_area_and_structural_gates():
    big_chair = make_object("chair", area_frac=0.05, object_index=0)
    tiny_chair = make_object("chair", area_frac=0.001, object_index=1, eligible=False)
    resolved = [big_chair, tiny_chair]

    # eligible_concept_counts only sees the big one -> looks single-instance...
    assert scene_objects.eligible_concept_counts(resolved)["chair"] == 1
    # ...but a human would see two chairs in the room, so the "single
    # instance" gate used by relative_depth/nearest_object/left_right must
    # see both.
    assert scene_objects.true_instance_counts(resolved)["chair"] == 2


# ------------------------------------------------------------- depth_utils.py

def test_backproject_maps_the_principal_point_to_the_camera_axis():
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    x, y, z = backproject_to_camera_frame(320.0, 240.0, 2.0, intrinsics)
    assert x == 0.0 and y == 0.0 and z == 2.0


def test_backproject_scales_correctly_with_depth():
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    x, y, _ = backproject_to_camera_frame(320.0 + 500.0, 240.0, 2.0, intrinsics)
    # one focal-length offset in pixels at depth=2m -> 2m of real-world x
    assert math.isclose(x, 2.0)
    assert y == 0.0
