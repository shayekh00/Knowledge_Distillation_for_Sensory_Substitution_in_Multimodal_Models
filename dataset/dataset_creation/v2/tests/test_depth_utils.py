"""load_intrinsics_file's two supported formats (depth_utils.py): SUN RGB-D's
9-value `intrinsics.txt` (a flattened 3x3 matrix) and ARKitScenes' 6-value
`.pincam` (`width height fx fy cx cy`) — the format the arkitscenes_plan.md
§6 Phase 3 investigation found silently unhandled (every `.pincam` file was
read as a 0-length-mismatch and treated as MISSING_INTRINSICS, dropping 85%
of nearest_object candidates for reasons that had nothing to do with real
data scarcity).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth_utils import load_intrinsics_file  # noqa: E402


def write(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def test_nine_value_intrinsics_txt_reshapes_to_3x3(tmp_path):
    path = os.path.join(tmp_path, "intrinsics.txt")
    write(path, "500 0 320 0 500 240 0 0 1")

    K = load_intrinsics_file(path)

    assert K.shape == (3, 3)
    np.testing.assert_array_equal(K, [[500, 0, 320], [0, 500, 240], [0, 0, 1]])


def test_six_value_pincam_builds_the_same_shaped_K_matrix(tmp_path):
    path = os.path.join(tmp_path, "40777073_1108.287.pincam")
    write(path, "256 192 211.5 211.5 127.8 95.4")

    K = load_intrinsics_file(path)

    assert K.shape == (3, 3)
    np.testing.assert_array_equal(K, [[211.5, 0, 127.8], [0, 211.5, 95.4], [0, 0, 1]])


def test_missing_file_returns_none(tmp_path):
    assert load_intrinsics_file(os.path.join(tmp_path, "absent.txt")) is None


def test_wrong_token_count_returns_none(tmp_path):
    path = os.path.join(tmp_path, "malformed.txt")
    write(path, "1 2 3 4 5")  # neither 6 nor 9 values

    assert load_intrinsics_file(path) is None
