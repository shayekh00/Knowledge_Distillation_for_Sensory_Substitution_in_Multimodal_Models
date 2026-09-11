"""Per-frame display rotation for ARKitScenes frames.

ARKitScenes' `lowres_wide`/`lowres_depth` frames are stored in the camera's
raw sensor orientation, not the "upright" orientation a human (or a person
viewing the audit tool) expects — the phone/iPad could be held in any of 4
physical orientations during capture, and which one applies is recorded
implicitly in the frame's own camera pose, not as a fixed per-scan or
per-dataset constant. This is not a guess: it is Apple's own documented
convention, vendored at
`dataset/dataset_creation/arkit_tools/ARKitScenes/threedod/benchmark_scripts/rectify_im.py`
(`decide_pose`/`rotate_pose`), which this module reimplements against the
project's own already-parsed pose representation
(`phase0_probe.load_traj`'s `{timestamp: (R, t)}`, R camera-to-world) rather
than re-deriving Apple's 4x4-matrix convention.

Scope note: this fixes *display* only (the audit viewer's image + polygon
overlay). It intentionally does not touch `data/index/scene_index_arkit.jsonl`,
the frozen v1.0 release, or the training/inference image-loading path
(`distillation/train_student.py`/`evaluation/zero_shot_inference.py`'s
`build_image`, which loads ARKit RGB with no rotation at all) — those feed
every real training and eval run, not just this reviewer tool, and fixing
them means recomputing geometry (`centroid_x/y`, `polygon_xy`, and anything
`left_right.py` derives from it) in the rotated frame throughout the
indexer, not just rotating pixels. That is a separate, larger fix, flagged
2026-09-11, not yet done.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Apple's z_orien table (rectify_im.py's decide_pose): index 0/1/2/3 =
# upright / rotate-90-CW / rotate-180 / rotate-90-CCW.
_Z_ORIEN = np.array([
    [0.0, -1.0, 0.0],   # upright
    [-1.0, 0.0, 0.0],   # left  (rotate 90 CW to correct)
    [0.0, 1.0, 0.0],    # upside-down (rotate 180 to correct)
    [1.0, 0.0, 0.0],    # right (rotate 90 CCW to correct)
])

_traj_cache: dict[Path, dict[str, tuple[np.ndarray, np.ndarray]]] = {}


def _load_traj(traj_path: Path) -> dict[str, np.ndarray]:
    """Parses `.traj` into `{timestamp: Rt}`, `Rt` the **camera-to-world**
    3x3 rotation block `decide_pose` expects (its own `z_orien` table is
    calibrated against that convention, per `rectify_im.py`'s worked
    examples).

    This is *not* the same matrix `phase0_probe.load_traj` returns for the
    same file, despite both starting from `Rotation.from_rotvec(rvec).
    as_matrix()` on the same rotation column: Apple's own
    `TrajStringToMatrix` names that raw matrix `r_w_to_p` (world-to-phone)
    and only reaches camera-to-world by assembling the full 4x4 extrinsics
    and inverting it (`Rt = np.linalg.inv(extrinsics)`) -- a step
    `phase0_probe.load_traj` skips, so its R is `decide_pose`'s convention
    transposed. Found by reproducing `rectify_im.py`'s own worked examples
    and getting a different orientation index than feeding it
    `phase0_probe.load_traj`'s R directly (2026-09-11) -- replicated here
    exactly (invert the full extrinsics) rather than guessing which
    docstring's "camera-to-world"/"world-to-camera" naming is the stale
    one, since `phase0_probe.py`'s own projection is the one empirically
    validated (Phase 0's visual box-alignment check).

    Cached per scan so repeated audit-item navigation doesn't re-parse the
    whole file every request.
    """
    if traj_path in _traj_cache:
        return _traj_cache[traj_path]
    from scipy.spatial.transform import Rotation
    poses: dict[str, np.ndarray] = {}
    with traj_path.open() as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            rvec = np.array([float(x) for x in parts[1:4]])
            tvec = np.array([float(x) for x in parts[4:7]])
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
            extrinsics[:3, 3] = tvec
            poses[parts[0]] = np.linalg.inv(extrinsics)[:3, :3]
    _traj_cache[traj_path] = poses
    return poses


def rotation_index_for(rgb_path: Path, frame_timestamp: str) -> int:
    """0/1/2/3 = upright / rotate-90-CW / rotate-180 / rotate-90-CCW needed
    to display this frame upright, per Apple's own pose-derived convention.

    Returns 0 (no rotation) if `rgb_path` isn't an ARKitScenes path or the
    matching `.traj` can't be found/parsed — this must never be the reason
    an audit item fails to load; SUN-RGB-D images are simply never rotated.
    """
    if "ARKitScenes" not in rgb_path.parts:
        return 0
    # rgb_path = .../<video_id>_frames/lowres_wide/<video_id>_<ts>.png
    traj_path = rgb_path.parent.parent / "lowres_wide.traj"
    if not traj_path.is_file():
        return 0
    try:
        poses = _load_traj(traj_path)
        target = float(frame_timestamp)
        closest_key = min(poses, key=lambda key: abs(float(key) - target))
        R = poses[closest_key]
    except (ValueError, OSError):
        return 0
    z_vec = R[2, :3]
    return int(np.argmax(_Z_ORIEN @ z_vec))


def rotate_image(image, rotation_index: int):
    """Apply the display rotation to a PIL Image. PIL's rotate angles are
    counter-clockwise, so Apple's clockwise convention needs the sign
    flipped; `expand=True` swaps width/height for the 90/270 cases."""
    if rotation_index == 0:
        return image
    degrees_ccw = {1: -90, 2: 180, 3: 90}[rotation_index]
    return image.rotate(degrees_ccw, expand=True)


def rotate_points(x: list[float], y: list[float], width: int, height: int,
                  rotation_index: int) -> tuple[list[float], list[float]]:
    """Transform polygon vertices the same way `rotate_image` transforms the
    W x H image they were computed against, so the overlay stays aligned."""
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if rotation_index == 0:
        new_x, new_y = x_arr, y_arr
    elif rotation_index == 1:  # 90 CW: new image is height x width
        new_x, new_y = height - y_arr, x_arr
    elif rotation_index == 2:  # 180: size unchanged
        new_x, new_y = width - x_arr, height - y_arr
    else:  # 3: 90 CCW: new image is height x width
        new_x, new_y = y_arr, width - x_arr
    return new_x.tolist(), new_y.tolist()


def rotated_dimensions(width: int, height: int, rotation_index: int) -> tuple[int, int]:
    return (height, width) if rotation_index in (1, 3) else (width, height)
