"""
P0 of the VQA-ARKitScenes-v1 pipeline (arkitscenes_plan.md §2, §6 Phase 1).

Builds one record **per sampled frame** (not per scan): paths, sensor, split,
and per-object geometry/depth statistics derived by **projecting each scan's
3D oriented boxes into that frame**, since ARKitScenes annotates in 3D/world
coordinates rather than per-image 2D masks (plan §3). Emits exactly the same
schema `build_index.py` (P0 for SUN RGB-D) does, so every downstream P1-P3
script and every question generator (P2) needs zero changes — the one
schema addition, `intrinsics_path`, is additive: SUN RGB-D records simply
lack it and `nearest_object.py` falls back to its old behaviour when it does
(`depth_utils.load_intrinsics`).

Reuses `dataset/dataset_creation/arkit_tools/phase0_probe.py`'s projection
and occlusion code directly rather than re-deriving it — that module is
already validated against 5 real scans (arkitscenes_plan.md §6, Phase 0
run 2026-09-08), including the near-distance gate for the perspective-
divergence failure mode found there.

**Per-frame object list semantics (the one real design decision this script
makes that `build_index.py` did not have to):** ARKitScenes' box annotations
are scene-level, valid across an entire ~500-2000-frame video, not per-image
like SUN RGB-D's. An object absent from a given frame's `objects` list here
means "not visible in this frame at all" (occluded, out of the frustum, or
behind the near-distance gate) — it is **omitted entirely**, not included
with `is_valid_polygon=False`, so that `scene_objects.present_concepts_any()`
correctly does not treat it as present for existence-negative safety (a scene-
wide-but-not-here object must not block a legitimate "is there an X" = "no"
question about this specific frame). An object that *is* visible but whose
projected geometry is too degenerate to trust (near-zero clipped area) still
gets `is_valid_polygon=False`, mirroring SUN RGB-D's INVALID_POLYGON case —
here the object is present, just not usable as a reference.

Usage::

    python dataset/dataset_creation/v2/build_index_arkit.py \\
        --scans-dir dataset/ARKitScenes_phase0/3dod/Training --split train \\
        --out data/index/scene_index_arkit.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
DATA_DIR = os.path.join(REPO_ROOT, "data")
BUILD_LOG_DIR = os.path.join(REPO_ROOT, "build_log")

sys.path.insert(0, os.path.join(REPO_ROOT, "dataset", "dataset_creation", "arkit_tools"))
from phase0_probe import (  # noqa: E402
    MIN_CORNER_DEPTH_M,
    load_intrinsics_pincam,
    load_traj,
    nearest_timestamp,
    obb_corners_world,
    project_points,
    world_to_camera,
)

from shapely.geometry import Polygon, box as shapely_box  # noqa: E402

SENSOR_NAME = "lidar"
DEPTH_MM_TO_M = 1000.0
OCCLUSION_MARGIN_M = 0.3  # phase0_probe.py's own threshold, kept identical

DROP_REASON = {
    "FRAME_MISSING_ASSET": "FRAME_MISSING_ASSET",
    "NO_POSE": "NO_POSE",
    "NO_OBJECTS_IN_VIEW": "NO_OBJECTS_IN_VIEW",
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def sample_frame_timestamps(intrinsics_dir: str, frames_per_scan: int) -> list:
    files = sorted(os.listdir(intrinsics_dir))
    if len(files) <= frames_per_scan:
        chosen = files
    else:
        stride = len(files) / frames_per_scan
        chosen = [files[int(i * stride)] for i in range(frames_per_scan)]
    prefix_len = len(files[0].rsplit("_", 1)[0]) + 1 if files else 0
    return [f[prefix_len:-len(".pincam")] for f in chosen]


def project_and_score_object(obj: dict, R_cam_to_world: np.ndarray, t_cam_in_world: np.ndarray,
                              K: np.ndarray, width: int, height: int,
                              depth_m: np.ndarray) -> dict | None:
    """Returns a fully-populated object record, or None if the object is not
    visible in this frame at all (should be omitted, not marked invalid)."""
    obb = obj["segments"]["obbAligned"]
    corners_world = obb_corners_world(
        np.array(obb["centroid"]), np.array(obb["axesLengths"]), np.array(obb["normalizedAxes"]))
    corners_camera = world_to_camera(corners_world, R_cam_to_world, t_cam_in_world)

    if np.all(corners_camera[:, 2] <= 0):
        return None
    in_front = corners_camera[corners_camera[:, 2] > 0]
    if in_front[:, 2].min() < MIN_CORNER_DEPTH_M:
        return None  # the near-distance failure mode found in Phase 0

    pixels, _ = project_points(in_front, K)
    valid = ~np.isnan(pixels).any(axis=1)
    pixels = pixels[valid]
    if pixels.shape[0] < 3:
        return None

    hull = Polygon(pixels).convex_hull
    clipped = hull.intersection(shapely_box(0, 0, width, height))
    if clipped.is_empty or clipped.area < 1.0:
        return None  # projects fully outside the frame: not visible here

    minx, miny, maxx, maxy = clipped.bounds
    mask_depth = depth_m[int(max(0, miny)):int(min(height, maxy)),
                         int(max(0, minx)):int(min(width, maxx))]
    valid_depth_px = mask_depth[mask_depth > 0]
    depth_valid_frac = float(valid_depth_px.size) / float(max(mask_depth.size, 1))
    observed_median = float(np.median(valid_depth_px)) if valid_depth_px.size else None
    box_depth_camera = float(np.median(in_front[:, 2]))

    occluded = observed_median is not None and observed_median < box_depth_camera - OCCLUSION_MARGIN_M
    if occluded:
        return None  # blocked by something nearer: not visible here either

    area_px = clipped.area
    centroid = clipped.centroid
    min_x, min_y, max_x, max_y = clipped.bounds
    touches_border = (min_x <= 2.0 or min_y <= 2.0
                      or max_x >= width - 2.0 or max_y >= height - 2.0)

    return {
        "raw_name": obj["label"],
        "is_valid_polygon": True,
        "area_px": float(area_px),
        "area_frac": float(area_px / (width * height)),
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
        "depth_median_m": observed_median,
        "depth_valid_frac": depth_valid_frac,
        "touches_border": touches_border,
        # The clipped 2D hull itself, not just its derived stats — SUN RGB-D
        # generators needing a real polygon (left_right.py's IoU overlap
        # gate) rebuild one from the raw annotation JSON, which has no
        # ARKitScenes equivalent; storing it here lets that generator use it
        # directly instead (arkitscenes_plan.md §6 Phase 3).
        "polygon_xy": [[float(x), float(y)] for x, y in clipped.exterior.coords],
    }


def process_one_frame(video_id: str, frame_timestamp: str, frames_dir: str,
                      annotation: dict, poses: dict, pose_timestamps: list,
                      drop_rows: list) -> dict | None:
    rgb_path = os.path.join(frames_dir, "lowres_wide", f"{video_id}_{frame_timestamp}.png")
    depth_path = os.path.join(frames_dir, "lowres_depth", f"{video_id}_{frame_timestamp}.png")
    intrinsics_path = os.path.join(
        frames_dir, "lowres_wide_intrinsics", f"{video_id}_{frame_timestamp}.pincam")
    image_id = f"{video_id}/{frame_timestamp}"

    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)
           and os.path.exists(intrinsics_path)):
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                          "reason_code": DROP_REASON["FRAME_MISSING_ASSET"],
                          "detail": f"{rgb_path} / {depth_path} / {intrinsics_path}"})
        return None

    nearest_pose_ts = nearest_timestamp(pose_timestamps, float(frame_timestamp))
    if abs(float(nearest_pose_ts) - float(frame_timestamp)) > 0.05:
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                          "reason_code": DROP_REASON["NO_POSE"], "detail": frame_timestamp})
        return None
    R_cam_to_world, t_cam_in_world = poses[nearest_pose_ts]

    K, width, height = load_intrinsics_pincam(intrinsics_path)
    with Image.open(rgb_path) as rgb_image:
        real_width, real_height = rgb_image.size
    depth_m = np.array(Image.open(depth_path), dtype=np.uint16).astype(np.float32) / DEPTH_MM_TO_M

    object_records = []
    for object_index, obj in enumerate(annotation["data"]):
        record = project_and_score_object(
            obj, R_cam_to_world, t_cam_in_world, K, real_width, real_height, depth_m)
        if record is None:
            continue  # not visible in this frame: omitted, see module docstring
        record["object_index"] = object_index
        object_records.append(record)

    if not object_records:
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                          "reason_code": DROP_REASON["NO_OBJECTS_IN_VIEW"], "detail": ""})
        return None

    return {
        "image_id": image_id,
        "sensor": SENSOR_NAME,
        "scene_type": "unknown",  # ARKitScenes' 3DOD metadata carries no room-type field
        "image_width": real_width,
        "image_height": real_height,
        "rgb_path": os.path.relpath(rgb_path, DATASET_DIR),
        "depth_path": os.path.relpath(depth_path, DATASET_DIR),
        "annotation_path": None,  # filled in by the caller (shared per scan)
        "intrinsics_path": os.path.relpath(intrinsics_path, DATASET_DIR),
        "objects": object_records,
        "sequence_id": video_id,
    }


def process_one_scan(scan_dir: str, video_id: str, split: str, frames_per_scan: int,
                     drop_rows: list) -> list:
    frames_dir = os.path.join(scan_dir, f"{video_id}_frames")
    annotation_path = os.path.join(scan_dir, f"{video_id}_3dod_annotation.json")
    intrinsics_dir = os.path.join(frames_dir, "lowres_wide_intrinsics")
    traj_path = os.path.join(frames_dir, "lowres_wide.traj")

    if not (os.path.isdir(intrinsics_dir) and os.path.isfile(annotation_path)
           and os.path.isfile(traj_path)):
        drop_rows.append({"image_id": video_id, "object_index": None, "raw_name": None,
                          "reason_code": DROP_REASON["FRAME_MISSING_ASSET"], "detail": scan_dir})
        return []

    with open(annotation_path) as handle:
        annotation = json.load(handle)
    poses = load_traj(traj_path)
    pose_timestamps = list(poses.keys())

    records = []
    for frame_timestamp in sample_frame_timestamps(intrinsics_dir, frames_per_scan):
        record = process_one_frame(video_id, frame_timestamp, frames_dir, annotation,
                                   poses, pose_timestamps, drop_rows)
        if record is None:
            continue
        record["annotation_path"] = os.path.relpath(annotation_path, DATASET_DIR)
        record["split"] = split
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scans-dir", required=True,
                        help="Directory of extracted scans, e.g. "
                             "dataset/ARKitScenes_phase0/3dod/Training")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"],
                        help="Value written to every record's `split` field. This script "
                             "does not itself decide train/val/test membership — that is a "
                             "Phase 3 policy decision (arkitscenes_plan.md), not resolved "
                             "here; caller passes whichever value this batch of scans "
                             "belongs to.")
    parser.add_argument("--frames-per-scan", type=int, default=8,
                        help="Well-spaced frames sampled per scan (plan §5: 5-10).")
    parser.add_argument("--config", default=os.path.join(DATA_DIR, "config.yaml"))
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "index", "scene_index_arkit.jsonl"))
    args = parser.parse_args()

    config = load_config(args.config)  # loaded for parity with build_index.py; thresholds
    _ = config                         # (min_area_frac etc.) are applied downstream in P1/P2

    scan_ids = sorted(d for d in os.listdir(args.scans_dir)
                      if os.path.isdir(os.path.join(args.scans_dir, d)))
    drop_rows: list = []
    all_records = []
    for video_id in scan_ids:
        scan_dir = os.path.join(args.scans_dir, video_id)
        records = process_one_scan(scan_dir, video_id, args.split, args.frames_per_scan, drop_rows)
        all_records.extend(records)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(BUILD_LOG_DIR, exist_ok=True)
    with open(args.out, "w") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")

    import pandas as pd
    pd.DataFrame(drop_rows).to_csv(
        os.path.join(BUILD_LOG_DIR, "p0_arkit_drops.csv"), index=False)

    label_counts = Counter(obj["raw_name"] for record in all_records for obj in record["objects"])
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "dataset/dataset_creation/v2/build_index_arkit.py",
        "scans_dir": os.path.relpath(args.scans_dir, REPO_ROOT),
        "split": args.split,
        "frames_per_scan": args.frames_per_scan,
        "counts": {
            "scans": len(scan_ids),
            "frames_kept": len(all_records),
            "frames_dropped": len({row["image_id"] for row in drop_rows}),
            "object_label_counts": dict(label_counts),
        },
    }
    manifest_path = os.path.join(os.path.dirname(args.out),
                                 f"manifest_arkit_{args.split}.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"{len(scan_ids)} scans -> {len(all_records)} frame records "
          f"({len(drop_rows)} drop rows) -> {args.out}")
    print(f"Object label counts: {dict(label_counts)}")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
