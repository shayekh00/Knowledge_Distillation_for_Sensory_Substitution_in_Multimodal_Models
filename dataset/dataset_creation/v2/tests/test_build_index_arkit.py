"""build_index_arkit.py's per-object projection/occlusion/gating logic,
on synthetic geometry with known answers — the Phase 1 deliverable
arkitscenes_plan.md §6 calls for, plus a regression test for the one real
failure mode Phase 0 found (a corner too close to the camera producing a
geometrically meaningless hull).

All tests use a camera at the world origin with identity rotation, so
`world_to_camera` is the identity and the arithmetic below is checkable by
hand: `pixel = fx * (x / z) + cx` etc.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_index_arkit import project_and_score_object  # noqa: E402

IDENTITY_ROTATION = np.eye(3)
CAMERA_AT_ORIGIN = np.zeros(3)
# fx=fy=100, cx=cy=50, so a 100x100-pixel frame is centred on the optical axis.
K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
WIDTH = HEIGHT = 100


def make_box(centroid, half_size=0.5, label="chair"):
    """A synthetic annotation object: an axis-aligned cube of side
    `2*half_size`, centred at `centroid`, in the same frame as the camera
    (normalizedAxes = identity — the box's local axes are the world axes)."""
    return {
        "label": label,
        "segments": {"obbAligned": {
            "centroid": list(centroid),
            "axesLengths": [2 * half_size] * 3,
            "normalizedAxes": list(IDENTITY_ROTATION.flatten()),
        }},
    }


def uniform_depth(value_m: float) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), value_m * 1000.0 / 1000.0, dtype=np.float32)


def test_a_box_directly_in_front_projects_near_image_centre():
    """Box centred on-axis at 2m: by symmetry its projected hull must be
    centred on the principal point (50, 50), and its area must be positive
    but well under the full frame."""
    box = make_box(centroid=(0.0, 0.0, 2.0), half_size=0.5)
    depth_m = uniform_depth(2.0)  # matches the box's own depth: not occluded

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    assert record is not None
    assert record["is_valid_polygon"] is True
    assert record["centroid_x"] == pytest.approx(50.0, abs=1.0)
    assert record["centroid_y"] == pytest.approx(50.0, abs=1.0)
    assert 0.0 < record["area_frac"] < 1.0
    assert record["depth_median_m"] == pytest.approx(2.0, abs=0.01)


def test_a_box_wholly_behind_the_camera_is_excluded():
    box = make_box(centroid=(0.0, 0.0, -2.0), half_size=0.5)
    depth_m = uniform_depth(2.0)

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    assert record is None


def test_occluded_object_is_excluded_not_flagged():
    """The negative control: a box at 2m whose projected footprint's
    observed LiDAR depth is much closer (0.5m) must be excluded outright —
    per the module's own documented semantics, an occluded object is
    **omitted**, not returned with `is_valid_polygon=False` (so
    `present_concepts_any()` never treats a not-actually-visible object as
    present in this frame)."""
    box = make_box(centroid=(0.0, 0.0, 2.0), half_size=0.5)
    depth_m = uniform_depth(0.5)  # something much nearer blocks the view

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    assert record is None


def test_object_at_matching_depth_is_not_occluded():
    """Same box, but the observed depth agrees with the box's own depth
    (within the occlusion margin) — must NOT be excluded. Pairing this with
    the previous test pins the boundary, not just one side of it."""
    box = make_box(centroid=(0.0, 0.0, 2.0), half_size=0.5)
    depth_m = uniform_depth(1.9)  # 0.1m nearer: well inside the 0.3m margin

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    assert record is not None


def test_a_corner_too_close_to_the_camera_is_excluded():
    """Regression test for the Phase 0 finding (arkitscenes_plan.md §6): a
    box with any corner nearer than MIN_CORNER_DEPTH_M produces wild
    perspective divergence (dividing by a near-zero z) and a geometrically
    meaningless hull, even though every corner is technically in front of
    the camera. A box centred at 0.3m with half_size 0.5 has a near face at
    z=-0.2 (behind — excluded by that check) and, with a smaller half_size
    that keeps every corner positive, still has corners well under 0.4m."""
    box = make_box(centroid=(0.0, 0.0, 0.3), half_size=0.1)  # z in [0.2, 0.4]
    depth_m = uniform_depth(0.3)

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    assert record is None


def test_a_box_that_projects_fully_outside_the_frame_is_excluded():
    """Off to the side and far enough that its projection never reaches
    the visible rectangle at all."""
    box = make_box(centroid=(50.0, 0.0, 2.0), half_size=0.5)
    depth_m = uniform_depth(2.0)

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    assert record is None


def test_object_record_matches_sun_rgbd_schema_keys():
    """Field-for-field parity with build_index.py's per-object record shape
    (plus `object_index`, added by the caller) — this is what lets every
    downstream P1-P3 script and P2 generator run unmodified against either
    dataset's index."""
    box = make_box(centroid=(0.0, 0.0, 2.0), half_size=0.5)
    depth_m = uniform_depth(2.0)

    record = project_and_score_object(
        box, IDENTITY_ROTATION, CAMERA_AT_ORIGIN, K, WIDTH, HEIGHT, depth_m)

    expected_keys = {
        "raw_name", "is_valid_polygon", "area_px", "area_frac",
        "centroid_x", "centroid_y", "depth_median_m", "depth_valid_frac",
        "touches_border",
    }
    assert expected_keys.issubset(record.keys())
