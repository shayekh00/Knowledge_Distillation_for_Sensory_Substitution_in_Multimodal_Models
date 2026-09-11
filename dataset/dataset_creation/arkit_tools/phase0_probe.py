"""Phase 0 feasibility probe (arkitscenes_plan.md §6, Phase 0).

Verifies, against 5 real downloaded scans, the four things the plan's gate
depends on:
  1. pose (.traj) and intrinsics (.pincam) parse into a usable form
  2. a projected 3D box lands where the object visibly is
  3. the occlusion test (depth-consistency) separates visible from occluded
  4. LiDAR depth decodes to plausible metres

Also resolves one ambiguity the plan did not anticipate: each annotation has
two OBB blocks (`segments.obb` and `segments.obbAligned`), at two different
scales/frames. `obbAligned` is meters, room-scale, and shares its coordinate
frame with the camera poses (verified below); `obb` is ~100x larger and is a
different (mesh-native) space entirely. `obbAligned` is the one to project.

Usage::

    python dataset/dataset_creation/arkit_tools/phase0_probe.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, box as shapely_box

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCANS_DIR = os.path.join(PROJECT_ROOT, "dataset", "ARKitScenes_phase0", "3dod", "Training")
MIN_CORNER_DEPTH_M = 0.4


def load_intrinsics_pincam(path: str) -> tuple[np.ndarray, int, int]:
    """.pincam: one line, `width height fx fy cx cy`. Returns the 3x3 K
    matrix `depth_utils.load_intrinsics` expects, plus width/height."""
    with open(path) as handle:
        width, height, fx, fy, cx, cy = (float(x) for x in handle.read().split())
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, int(width), int(height)


def load_traj(path: str) -> dict:
    """.traj: one line per frame, `timestamp rx ry rz tx ty tz` (axis-angle
    radians, translation metres). Returns {timestamp_str: (R, t)}.

    `R`/`t` are the **world-to-camera** transform (`p_cam = R @ p_world + t`),
    matching what Apple's own `TrajStringToMatrix` calls `r_w_to_p`/`t_w_to_p`
    before it inverts them. The docstring here previously claimed the opposite
    (camera-to-world); that error propagated into `world_to_camera` and
    silently mis-projected every ARKitScenes box — see that function's own
    note, corrected 2026-09-11."""
    poses = {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            timestamp = parts[0]
            rvec = np.array([float(x) for x in parts[1:4]])
            tvec = np.array([float(x) for x in parts[4:7]])
            R = Rotation.from_rotvec(rvec).as_matrix()
            poses[timestamp] = (R, tvec)
    return poses


def world_to_camera(points_world: np.ndarray, R_world_to_cam: np.ndarray,
                    t_world_to_cam: np.ndarray) -> np.ndarray:
    """World point -> camera-frame coordinates: `R @ p_world + t`.

    **Corrected 2026-09-11.** This previously read `R^T @ (p_world - t)`, on
    the docstring's claim that `.traj`'s rotation column is a *camera-to-world*
    pose. It is not. Apple's own loader builds the very same matrix from the
    same angle-axis column and names it `r_w_to_p` — world-to-phone — then
    assembles `extrinsics = [R|t]` and only reaches camera-to-world by
    inverting the whole 4x4 (`Rt = np.linalg.inv(extrinsics)`,
    `ARKitScenes/threedod/benchmark_scripts/utils/tenFpsDataLoader.py`
    `TrajStringToMatrix`). So `R` here is already world-to-camera and must be
    applied directly, not transposed.

    Measured on a real frame (Validation/42898862, ts 196391.024, 32 annotated
    boxes) the old form put 51 of the projected corners inside the image and
    left every object partially clipped against x=0; the corrected form puts
    92 inside with objects landing cleanly 8-of-8, and the rendered hulls sit
    on the actual furniture instead of piling into a strip down the left edge.

    Not caught earlier because every test in
    `dataset/dataset_creation/v2/tests/test_build_index_arkit.py` places the
    camera at the world origin with identity rotation, where `R^T @ (p - t)`
    and `R @ p + t` are the same function — see that module's own docstring.
    """
    return (R_world_to_cam @ points_world.T).T + t_world_to_cam


def obb_corners_world(centroid: np.ndarray, axes_lengths: np.ndarray,
                      normalized_axes: np.ndarray) -> np.ndarray:
    """8 corners of an oriented box from ARKitScenes' `obbAligned` fields.

    `normalizedAxes` is a row-major flattened 3x3: each ROW is one of the
    box's three local axis directions (unit vectors) in world frame, scaled
    by half of the matching `axesLengths` entry to reach a corner offset.
    """
    R = normalized_axes.reshape(3, 3)
    half = axes_lengths / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    offsets = signs * half  # [8, 3] in the box's own local axis units
    corners = centroid + offsets @ R  # each row: half*sign(i) applied along R's rows
    return corners


def project_points(points_camera: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection. Returns (pixel_xy [N,2], depth_z [N]); a point
    behind the camera has depth_z <= 0 and its pixel coordinates are not
    meaningful (caller must filter on depth_z first)."""
    z = points_camera[:, 2]
    x = points_camera[:, 0] / np.where(z != 0, z, np.nan)
    y = points_camera[:, 1] / np.where(z != 0, z, np.nan)
    pixel = (K[:2, :2] @ np.stack([x, y]) + K[:2, 2:3]).T
    return pixel, z


def nearest_timestamp(available: list, target: float) -> str:
    return min(available, key=lambda t: abs(float(t) - target))


def check_one_scan(scan_dir: str, video_id: str) -> dict:
    frames_dir = os.path.join(scan_dir, f"{video_id}_frames")
    traj_path = os.path.join(frames_dir, "lowres_wide.traj")
    intrinsics_dir = os.path.join(frames_dir, "lowres_wide_intrinsics")
    depth_dir = os.path.join(frames_dir, "lowres_depth")
    annotation_path = os.path.join(scan_dir, f"{video_id}_3dod_annotation.json")

    poses = load_traj(traj_path)
    with open(annotation_path) as handle:
        annotation = json.load(handle)

    intrinsics_files = sorted(os.listdir(intrinsics_dir))
    pose_timestamps = list(poses.keys())

    results = {"video_id": video_id, "n_frames": len(intrinsics_files),
              "n_boxes": len(annotation["data"]), "projections": []}

    # A handful of well-spaced frames, not all 560 — this is a probe, not the
    # indexer itself.
    sample_files = intrinsics_files[::max(1, len(intrinsics_files) // 8)][:8]

    for intrinsics_file in sample_files:
        frame_timestamp = intrinsics_file.replace(f"{video_id}_", "").replace(".pincam", "")
        K, width, height = load_intrinsics_pincam(os.path.join(intrinsics_dir, intrinsics_file))
        pose_ts = nearest_timestamp(pose_timestamps, float(frame_timestamp))
        if abs(float(pose_ts) - float(frame_timestamp)) > 0.05:
            continue  # no close-enough pose for this frame; skip rather than misattribute
        R_world_to_cam, t_world_to_cam = poses[pose_ts]

        depth_path = os.path.join(depth_dir, f"{video_id}_{frame_timestamp}.png")
        if not os.path.isfile(depth_path):
            continue
        depth_m = np.array(Image.open(depth_path), dtype=np.uint16).astype(np.float32) / 1000.0

        for obj in annotation["data"]:
            obb = obj["segments"]["obbAligned"]
            centroid = np.array(obb["centroid"])
            axes_lengths = np.array(obb["axesLengths"])
            normalized_axes = np.array(obb["normalizedAxes"])
            corners_world = obb_corners_world(centroid, axes_lengths, normalized_axes)
            corners_camera = world_to_camera(corners_world, R_world_to_cam, t_world_to_cam)

            if np.all(corners_camera[:, 2] <= 0):
                continue  # wholly behind the camera

            in_front = corners_camera[corners_camera[:, 2] > 0]
            # A corner within this distance of the camera projects with
            # extreme perspective divergence (pinhole division by a small z)
            # even when every corner is technically "in front": found on real
            # data (a chair box with corners at 0.18-1.6m produced a "clipped"
            # hull covering 30% of the frame that visually corresponded to no
            # part of the actual chair). Not a code bug — a real geometric
            # failure mode of corner-hull projection when a box has any
            # corner this close, so it is excluded here rather than trusted.
            if in_front[:, 2].min() < MIN_CORNER_DEPTH_M:
                continue
            pixels, _ = project_points(in_front, K)
            valid = ~np.isnan(pixels).any(axis=1)
            pixels = pixels[valid]
            if pixels.shape[0] < 3:
                continue  # not enough surviving corners for a hull

            try:
                hull = Polygon(pixels).convex_hull
            except Exception:
                continue
            image_rect = shapely_box(0, 0, width, height)
            clipped = hull.intersection(image_rect)
            if clipped.is_empty or clipped.area < 1.0:
                continue  # projects fully outside the frame

            minx, miny, maxx, maxy = clipped.bounds
            mask_depth = depth_m[int(max(0, miny)):int(min(height, maxy)),
                                 int(max(0, minx)):int(min(width, maxx))]
            valid_depth = mask_depth[mask_depth > 0]
            observed_median = float(np.median(valid_depth)) if valid_depth.size else None
            box_depth_camera = float(np.median(in_front[:, 2]))

            occluded = (observed_median is not None
                       and observed_median < box_depth_camera - 0.3)  # metres; §3's test

            results["projections"].append({
                "frame": frame_timestamp, "label": obj["label"],
                "area_frac": clipped.area / (width * height),
                "centroid_px": [clipped.centroid.x, clipped.centroid.y],
                "box_depth_camera_m": round(box_depth_camera, 3),
                "observed_median_depth_m": (round(observed_median, 3)
                                            if observed_median is not None else None),
                "occluded": occluded,
            })

    return results


def main() -> None:
    scan_ids = sorted(os.listdir(SCANS_DIR))
    all_results = []
    for video_id in scan_ids:
        scan_dir = os.path.join(SCANS_DIR, video_id)
        if not os.path.isdir(scan_dir):
            continue
        print(f"=== {video_id} ===")
        result = check_one_scan(scan_dir, video_id)
        print(f"  {result['n_frames']} frames, {result['n_boxes']} boxes, "
              f"{len(result['projections'])} object-frame projections landed in view")
        n_occluded = sum(1 for p in result["projections"] if p["occluded"])
        print(f"  {n_occluded}/{len(result['projections'])} flagged occluded")
        for p in result["projections"][:5]:
            print(f"    frame {p['frame']}  {p['label']!r:20s}  "
                  f"area_frac={p['area_frac']:.4f}  "
                  f"box_depth={p['box_depth_camera_m']}m  "
                  f"observed={p['observed_median_depth_m']}m  "
                  f"occluded={p['occluded']}")
        all_results.append(result)

    total_projections = sum(len(r["projections"]) for r in all_results)
    total_occluded = sum(sum(1 for p in r["projections"] if p["occluded"]) for r in all_results)
    print(f"\n=== summary: {total_projections} projections across {len(all_results)} scans, "
          f"{total_occluded} ({100 * total_occluded / total_projections:.1f}%) flagged occluded ===")


if __name__ == "__main__":
    main()
