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


def load_intrinsics_file(intrinsics_path: str) -> np.ndarray | None:
    """Read a 3x3 intrinsics matrix from an absolute path to either an
    `intrinsics.txt`-shaped file (9 whitespace-separated floats, SUN RGB-D)
    or a `.pincam`-shaped file (6 floats: `width height fx fy cx cy`,
    ARKitScenes — same format `arkit_tools/phase0_probe.py`'s
    `load_intrinsics_pincam` already parses; duplicated here rather than
    imported so this dataset-agnostic module has no dependency on that
    ARKitScenes-specific tool script). Factored out of `load_intrinsics` so
    a caller that already knows exactly which file it wants (e.g.
    ARKitScenes' `intrinsics_path` schema field, one real file per frame
    rather than one per scene) does not have to go through scene-directory
    reconstruction to get there — see `arkitscenes_plan.md` §2 on why that
    reconstruction is dataset-specific and not safe to generalize blindly."""
    if not os.path.exists(intrinsics_path):
        return None
    with open(intrinsics_path, "r") as intrinsics_file:
        values = [float(token) for token in intrinsics_file.read().split()]
    if len(values) == 9:
        return np.array(values, dtype=np.float64).reshape(3, 3)
    if len(values) == 6:
        _width, _height, fx, fy, cx, cy = values
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return None


def load_intrinsics(scene_dir_absolute: str) -> np.ndarray | None:
    """SUN RGB-D's convention: one `intrinsics.txt` per scene directory."""
    return load_intrinsics_file(os.path.join(scene_dir_absolute, "intrinsics.txt"))


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
