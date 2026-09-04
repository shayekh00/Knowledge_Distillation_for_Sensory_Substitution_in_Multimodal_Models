"""
Depth helpers shared by the P2 question generators.

decode_sunrgbd_depth() duplicates the small function in build_index.py
(P0) on purpose rather than importing it: P0 is already verified against
the real corpus (see docs/DATASET_CREATION_PLAN.md §2 status note) and is
left untouched to avoid any regression risk. The two copies must stay
identical; there are only four lines to keep in sync.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def decode_sunrgbd_depth(depth_path: str, clip_max_m: float) -> np.ndarray:
    raw = np.array(Image.open(depth_path), dtype=np.uint16)
    rotated = (raw >> 3) | (raw << 13).astype(np.uint16)
    depth_m = rotated.astype(np.float32) / 1000.0
    return np.clip(depth_m, 0.0, clip_max_m)


def load_intrinsics(scene_dir_absolute: str) -> np.ndarray | None:
    intrinsics_path = os.path.join(scene_dir_absolute, "intrinsics.txt")
    if not os.path.exists(intrinsics_path):
        return None
    with open(intrinsics_path, "r") as intrinsics_file:
        values = [float(token) for token in intrinsics_file.read().split()]
    if len(values) != 9:
        return None
    return np.array(values, dtype=np.float64).reshape(3, 3)


def backproject_to_camera_frame(pixel_x: float, pixel_y: float, depth_m: float,
                                 camera_intrinsics: np.ndarray) -> tuple:
    """
    Pinhole back-projection into the camera's own 3-D frame (no Rtilt
    world-alignment applied). That is fine here: every nearest-object
    comparison happens between two points from the *same* camera capture,
    and a shared rotation does not change the Euclidean distance between
    them, so skipping Rtilt is exact for this use, not an approximation.
    """
    focal_x, focal_y = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    principal_x, principal_y = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    x = (pixel_x - principal_x) * depth_m / focal_x
    y = (pixel_y - principal_y) * depth_m / focal_y
    return x, y, depth_m
