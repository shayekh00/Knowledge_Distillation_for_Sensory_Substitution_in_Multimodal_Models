"""
P0 of the VQA-SUNRGBD-v2 pipeline (see docs/DATASET_CREATION_PLAN.md).

Builds one clean, verified record per SUNRGBD scene: paths, sensor, scene
type, train/val/test split, per-object polygon geometry, and per-object
depth statistics. This script contains no question-generation logic —
that lives in the per-type generators (P2), which all read the JSONL
this script produces instead of touching SUNRGBDMeta.mat or annotation
JSON directly.

Split rule (Rule S1/S2 in the plan):
  - test  = the official SUNRGBD test split (SUNRGBDtoolbox allsplit.mat)
  - train/val = a sequence-grouped, seeded split of the official train pool,
    so that near-duplicate frames from the same recording never appear in
    both train and val.

Depth decoding follows the official SUNRGBD convention
(SUNRGBDtoolbox/readData/read3dPoints.m):
    depth_raw16 = (pixel >> 3) | (pixel << 13)   # 16-bit rotate
    depth_m     = depth_raw16 / 1000, clipped to [0, clip_max_m]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io
import yaml
from PIL import Image
from shapely.geometry import Polygon
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
SUNRGBD_DIR = os.path.join(DATASET_DIR, "SUNRGBD")
TOOLBOX_DIR = os.path.join(DATASET_DIR, "SUNRGBDtoolbox")
DATA_DIR = os.path.join(REPO_ROOT, "data")
BUILD_LOG_DIR = os.path.join(REPO_ROOT, "build_log")

SUNRGBD_META_PATH = os.path.join(TOOLBOX_DIR, "Metadata", "SUNRGBDMeta.mat")
ALLSPLIT_PATH = os.path.join(TOOLBOX_DIR, "traintestSUNRGBD", "allsplit.mat")

DROP_REASON = {
    "RGB_MISSING": "RGB_MISSING",
    "DEPTH_MISSING": "DEPTH_MISSING",
    "ANNOTATION_MISSING": "ANNOTATION_MISSING",
    "ANNOTATION_UNPARSEABLE": "ANNOTATION_UNPARSEABLE",
    "NO_OBJECTS": "NO_OBJECTS",
    "INVALID_POLYGON": "INVALID_POLYGON",
    "SPLIT_UNASSIGNED": "SPLIT_UNASSIGNED",
    "SEQUENCE_SHARED_WITH_TEST": "SEQUENCE_SHARED_WITH_TEST",
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def md5_of_file(path: str) -> str:
    file_hash = hashlib.md5()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def strip_sunrgbd_prefix(absolute_path: str) -> str:
    marker = "/n/fs/sun3d/data/SUNRGBD/"
    if marker in absolute_path:
        return absolute_path.split(marker, 1)[1]
    marker_alt = "SUNRGBD/"
    return absolute_path.split(marker_alt, 1)[1]


def scene_dir_from_rgbpath(relative_rgbpath: str) -> str:
    # relative_rgbpath looks like "kv1/NYUdata/NYU0428/image/NYU0428.jpg"
    # (sometimes with a doubled slash before "image", e.g. NYUdata scenes:
    # "kv1/NYUdata/NYU0428//image/NYU0428.jpg" — normalize that away).
    parts = [part for part in relative_rgbpath.split("/") if part != ""]
    return "/".join(parts[:-2])


def load_official_split(allsplit_path: str) -> tuple[set, set]:
    # allsplit.mat entries carry a trailing slash ("kv1/NYUdata/NYU0003/");
    # our scene ids never do, so both sides must be normalized the same way.
    mat = scipy.io.loadmat(allsplit_path, squeeze_me=True, struct_as_record=False)
    train_paths = {strip_sunrgbd_prefix(path).rstrip("/") for path in mat["alltrain"]}
    test_paths = {strip_sunrgbd_prefix(path).rstrip("/") for path in mat["alltest"]}
    return train_paths, test_paths


def sequence_group_id(image_id: str, group_mode: str) -> str:
    parts = image_id.split("/")
    if group_mode == "building_room" and "sun3ddata" in parts:
        sun3d_index = parts.index("sun3ddata")
        building_and_room = parts[sun3d_index + 1 : sun3d_index + 3]
        if len(building_and_room) == 2:
            return "sun3d:" + "/".join(building_and_room)
    # Every other sensor family has exactly one capture per scene folder
    # (verified: 10,335 SUNRGBDMeta entries have 10,335 distinct
    # sequenceName values), so the scene itself is its own group.
    return "scene:" + image_id


def decode_sunrgbd_depth(depth_path: str, clip_max_m: float) -> np.ndarray:
    raw = np.array(Image.open(depth_path), dtype=np.uint16)
    rotated = (raw >> 3) | (raw << 13).astype(np.uint16)
    depth_m = rotated.astype(np.float32) / 1000.0
    return np.clip(depth_m, 0.0, clip_max_m)


BORDER_TOLERANCE_PX = 2.0


def touches_image_border(polygon: Polygon, image_width: int, image_height: int) -> bool:
    """True if the polygon's bounding box reaches the edge of the frame.

    A polygon is annotated only over what is visible, so we cannot measure
    how much of an object lies outside the frame — only that its silhouette
    is cut off exactly at the boundary, which is the signal an audited
    disagreement traced back to (P4: a reference object visible only as a
    sliver at the image edge, e.g. a cup, cannot be identified by a human
    or a model from that sliver alone, no matter how the question is
    worded). `min_area` alone does not catch this: a small sliver can still
    clear the area floor while being unidentifiable.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    return (
        min_x <= BORDER_TOLERANCE_PX
        or min_y <= BORDER_TOLERANCE_PX
        or max_x >= image_width - BORDER_TOLERANCE_PX
        or max_y >= image_height - BORDER_TOLERANCE_PX
    )


def rasterize_polygon_mask(polygon: Polygon, image_height: int, image_width: int) -> np.ndarray:
    from PIL import ImageDraw

    mask_image = Image.new("L", (image_width, image_height), 0)
    exterior_coords = [(x, y) for x, y in polygon.exterior.coords]
    ImageDraw.Draw(mask_image).polygon(exterior_coords, outline=1, fill=1)
    return np.array(mask_image, dtype=bool)


def build_polygon_records(annotation_data: dict, image_width: int, image_height: int,
                           depth_m: np.ndarray, config: dict, image_id: str,
                           drop_rows: list) -> list:
    object_names = [
        obj.get("name", "") if isinstance(obj, dict) else ""
        for obj in annotation_data.get("objects", [])
    ]
    frames = annotation_data.get("frames", [])
    polygons_by_object_index = defaultdict(list)
    if frames:
        for polygon_entry in frames[0].get("polygon", []):
            polygons_by_object_index[polygon_entry.get("object")].append(polygon_entry)

    image_area = float(image_width * image_height)
    object_records = []

    for object_index, raw_name in enumerate(object_names):
        polygon_entries = polygons_by_object_index.get(object_index, [])
        if not polygon_entries:
            object_records.append({
                "object_index": object_index,
                "raw_name": raw_name,
                "is_valid_polygon": False,
                "area_px": None, "area_frac": None,
                "centroid_x": None, "centroid_y": None,
                "depth_median_m": None, "depth_valid_frac": None,
                "touches_border": False,
            })
            continue

        # An object can have more than one polygon in the same frame
        # (e.g. an occluded object split into pieces); union them.
        shapely_polygons = []
        for polygon_entry in polygon_entries:
            xs, ys = polygon_entry.get("x", []), polygon_entry.get("y", [])
            if not isinstance(xs, list):
                xs = [xs]
            if not isinstance(ys, list):
                ys = [ys]
            points = list(zip(xs, ys))
            if len(points) < 3:
                continue
            candidate = Polygon(points).buffer(0)
            if not candidate.is_empty and candidate.area > 0:
                shapely_polygons.append(candidate)

        if not shapely_polygons:
            drop_rows.append({
                "image_id": image_id, "object_index": object_index,
                "raw_name": raw_name, "reason_code": DROP_REASON["INVALID_POLYGON"],
                "detail": "no polygon with >=3 valid points and positive area after repair",
            })
            object_records.append({
                "object_index": object_index,
                "raw_name": raw_name,
                "is_valid_polygon": False,
                "area_px": None, "area_frac": None,
                "centroid_x": None, "centroid_y": None,
                "depth_median_m": None, "depth_valid_frac": None,
                "touches_border": False,
            })
            continue

        union_polygon = shapely_polygons[0]
        for extra_polygon in shapely_polygons[1:]:
            union_polygon = union_polygon.union(extra_polygon)

        centroid = union_polygon.centroid
        area_px = union_polygon.area

        depth_median_m, depth_valid_frac = None, 0.0
        try:
            mask = rasterize_polygon_mask(union_polygon, image_height, image_width)
            masked_depth = depth_m[mask]
            valid_depth = masked_depth[masked_depth > 0]
            depth_valid_frac = float(len(valid_depth)) / float(max(len(masked_depth), 1))
            if depth_valid_frac >= config["depth"]["min_valid_fraction"]:
                depth_median_m = float(np.median(valid_depth))
        except Exception:
            pass

        object_records.append({
            "object_index": object_index,
            "raw_name": raw_name,
            "is_valid_polygon": True,
            "area_px": float(area_px),
            "area_frac": float(area_px / image_area) if image_area > 0 else None,
            "centroid_x": float(centroid.x),
            "centroid_y": float(centroid.y),
            "depth_median_m": depth_median_m,
            "depth_valid_frac": float(depth_valid_frac),
            "touches_border": touches_image_border(union_polygon, image_width, image_height),
        })

    return object_records


def process_one_scene(meta_entry, config: dict, drop_rows: list) -> dict | None:
    relative_rgbpath = strip_sunrgbd_prefix(meta_entry.rgbpath)
    image_id = scene_dir_from_rgbpath(relative_rgbpath)
    scene_dir = os.path.join(SUNRGBD_DIR, image_id)

    rgb_path = os.path.join(scene_dir, "image", meta_entry.rgbname)
    depth_path = os.path.join(scene_dir, "depth_bfx", meta_entry.depthname)
    annotation_path = os.path.join(scene_dir, "annotation", "index.json")

    if not os.path.exists(rgb_path):
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                           "reason_code": DROP_REASON["RGB_MISSING"], "detail": rgb_path})
        return None
    if not os.path.exists(depth_path):
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                           "reason_code": DROP_REASON["DEPTH_MISSING"], "detail": depth_path})
        return None
    if not os.path.exists(annotation_path):
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                           "reason_code": DROP_REASON["ANNOTATION_MISSING"], "detail": annotation_path})
        return None

    try:
        with open(annotation_path, "r") as annotation_file:
            annotation_data = json.load(annotation_file)
    except Exception as error:
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                           "reason_code": DROP_REASON["ANNOTATION_UNPARSEABLE"], "detail": str(error)})
        return None

    if not annotation_data.get("objects"):
        drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                           "reason_code": DROP_REASON["NO_OBJECTS"], "detail": "empty objects list"})
        return None

    with Image.open(rgb_path) as rgb_image:
        image_width, image_height = rgb_image.size

    depth_m = decode_sunrgbd_depth(depth_path, config["depth"]["clip_max_m"])

    scene_txt_path = os.path.join(scene_dir, "scene.txt")
    scene_type = "unknown"
    if os.path.exists(scene_txt_path):
        with open(scene_txt_path, "r") as scene_file:
            scene_type = scene_file.read().strip().lower() or "unknown"

    object_records = build_polygon_records(
        annotation_data, image_width, image_height, depth_m, config, image_id, drop_rows
    )

    return {
        "image_id": image_id,
        "sensor": str(meta_entry.sensorType),
        "scene_type": scene_type,
        "image_width": image_width,
        "image_height": image_height,
        "rgb_path": os.path.relpath(rgb_path, DATASET_DIR),
        "depth_path": os.path.relpath(depth_path, DATASET_DIR),
        "annotation_path": os.path.relpath(annotation_path, DATASET_DIR),
        "objects": object_records,
    }


def assign_split(image_ids: list, sequence_ids: list, train_pool: set, test_pool: set,
                  val_fraction: float, seed: int, drop_rows: list) -> dict:
    split_by_image_id = {}
    train_pool_rows = []

    # Rule S2b: the official test split is fixed, and 41 of its sun3d room
    # groups also contain official-train frames — near-duplicate keyframes of
    # the same room. Those frames are dropped from train/val rather than
    # touching the official test set, so no test image shares a capture
    # sequence with a training image while test stays comparable to every
    # other SUNRGBD paper. Test is never trimmed for this reason.
    test_sequence_ids = {
        sequence_id for image_id, sequence_id in zip(image_ids, sequence_ids)
        if image_id in test_pool
    }

    for image_id, sequence_id in zip(image_ids, sequence_ids):
        if image_id in test_pool:
            split_by_image_id[image_id] = "test"
        elif image_id in train_pool:
            if sequence_id in test_sequence_ids:
                drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                                   "reason_code": DROP_REASON["SEQUENCE_SHARED_WITH_TEST"],
                                   "detail": f"sequence {sequence_id} also has test frames"})
                continue
            train_pool_rows.append((image_id, sequence_id))
        else:
            drop_rows.append({"image_id": image_id, "object_index": None, "raw_name": None,
                               "reason_code": DROP_REASON["SPLIT_UNASSIGNED"],
                               "detail": "not present in official alltrain or alltest"})

    if train_pool_rows:
        pool_image_ids = [row[0] for row in train_pool_rows]
        pool_group_ids = [row[1] for row in train_pool_rows]
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_indices, val_indices = next(splitter.split(pool_image_ids, groups=pool_group_ids))
        for index in train_indices:
            split_by_image_id[pool_image_ids[index]] = "train"
        for index in val_indices:
            split_by_image_id[pool_image_ids[index]] = "val"

    return split_by_image_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P0 scene index for VQA-SUNRGBD-v2.")
    parser.add_argument("--config", default=os.path.join(DATA_DIR, "config.yaml"))
    parser.add_argument("--max_scenes", type=int, default=None, help="Debug: process only N scenes.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]

    print("Loading official SUNRGBD metadata and split...")
    meta_entries = scipy.io.loadmat(
        SUNRGBD_META_PATH, squeeze_me=True, struct_as_record=False
    )["SUNRGBDMeta"]
    train_pool, test_pool = load_official_split(ALLSPLIT_PATH)
    assert len(train_pool & test_pool) == 0, "official train/test paths overlap"

    if args.max_scenes:
        meta_entries = meta_entries[: args.max_scenes]

    drop_rows: list = []
    scene_records = []
    for meta_entry in tqdm(meta_entries, desc="Indexing scenes"):
        record = process_one_scene(meta_entry, config, drop_rows)
        if record is not None:
            scene_records.append(record)

    image_ids = [record["image_id"] for record in scene_records]
    sequence_ids = [
        sequence_group_id(image_id, config["split"]["sun3d_group_by"]) for image_id in image_ids
    ]
    split_by_image_id = assign_split(
        image_ids, sequence_ids, train_pool, test_pool,
        config["split"]["val_fraction"], seed, drop_rows,
    )

    kept_records = []
    for record, sequence_id in zip(scene_records, sequence_ids):
        split = split_by_image_id.get(record["image_id"])
        if split is None:
            continue
        record["split"] = split
        record["sequence_id"] = sequence_id
        kept_records.append(record)

    os.makedirs(os.path.join(DATA_DIR, "index"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "splits"), exist_ok=True)
    os.makedirs(BUILD_LOG_DIR, exist_ok=True)

    index_path = os.path.join(DATA_DIR, "index", "scene_index.jsonl")
    with open(index_path, "w") as index_file:
        for record in kept_records:
            index_file.write(json.dumps(record) + "\n")

    split_counts = Counter(record["split"] for record in kept_records)
    for split_name in ("train", "val", "test"):
        split_image_ids = sorted(
            record["image_id"] for record in kept_records if record["split"] == split_name
        )
        with open(os.path.join(DATA_DIR, "splits", f"{split_name}_images.txt"), "w") as split_file:
            split_file.write("\n".join(split_image_ids) + ("\n" if split_image_ids else ""))

    pd.DataFrame(drop_rows).to_csv(os.path.join(BUILD_LOG_DIR, "p0_drops.csv"), index=False)

    # Frozen plausibility basis for existence.py's hard negatives (Rule 4.1b:
    # "co-occurs with the scene's scene_type in >=5% of that scene type's
    # images"). Computed here, over every scene that parsed — including the
    # ones later dropped for sequence overlap — because which objects plausibly
    # belong in a bedroom is a fact about SUNRGBD, not about our split hygiene.
    # Committing it keeps test questions invariant when split decisions change;
    # recomputing it per run made them shift.
    scene_type_object_counts = defaultdict(Counter)
    scene_type_totals = Counter()
    for record in scene_records:
        scene_type_totals[record["scene_type"]] += 1
        raw_names = {obj["raw_name"] for obj in record["objects"]}
        scene_type_object_counts[record["scene_type"]].update(raw_names)
    cooccurrence = {
        scene_type: {
            raw_name: count / scene_type_totals[scene_type]
            for raw_name, count in counts.items()
        }
        for scene_type, counts in scene_type_object_counts.items()
    }
    cooccurrence_path = os.path.join(DATA_DIR, "vocab", "scene_type_cooccurrence.json")
    os.makedirs(os.path.dirname(cooccurrence_path), exist_ok=True)
    with open(cooccurrence_path, "w") as cooccurrence_file:
        json.dump({"scenes_per_type": dict(scene_type_totals), "raw_name_fraction": cooccurrence},
                   cooccurrence_file, indent=2, sort_keys=True)

    sensor_by_split = defaultdict(Counter)
    scene_type_by_split = defaultdict(Counter)
    for record in kept_records:
        sensor_by_split[record["split"]][record["sensor"]] += 1
        scene_type_by_split[record["split"]][record["scene_type"]] += 1

    sequence_group_sizes = Counter(sequence_ids)
    multi_member_groups = {group: size for group, size in sequence_group_sizes.items() if size > 1}

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "dataset/dataset_creation/v2/build_index.py",
        "config_path": os.path.relpath(args.config, REPO_ROOT),
        "config": config,
        "git_repo": False,
        "toolbox_checksums": {
            "SUNRGBDMeta.mat": md5_of_file(SUNRGBD_META_PATH),
            "allsplit.mat": md5_of_file(ALLSPLIT_PATH),
        },
        "counts": {
            "total_meta_entries": int(len(meta_entries)),
            "kept_scenes": len(kept_records),
            "dropped_scenes": len({row["image_id"] for row in drop_rows if row["object_index"] is None}),
            "split_sizes": dict(split_counts),
            "sensor_by_split": {split: dict(counter) for split, counter in sensor_by_split.items()},
            "scene_type_by_split": {split: dict(counter) for split, counter in scene_type_by_split.items()},
        },
        "sequence_grouping": {
            "total_groups": len(sequence_group_sizes),
            "groups_with_multiple_frames": len(multi_member_groups),
            "largest_group_size": max(sequence_group_sizes.values()) if sequence_group_sizes else 0,
        },
    }
    with open(os.path.join(DATA_DIR, "index", "manifest.json"), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"\nKept {len(kept_records)} / {len(meta_entries)} scenes.")
    print(f"Split sizes: {dict(split_counts)}")
    print(f"Sequence groups: {len(sequence_group_sizes)} total, "
          f"{len(multi_member_groups)} span multiple frames "
          f"(largest: {manifest['sequence_grouping']['largest_group_size']} frames).")
    print(f"Drop rows logged: {len(drop_rows)} -> build_log/p0_drops.csv")
    print(f"Index written: {index_path}")
    print(f"Manifest written: {os.path.join(DATA_DIR, 'index', 'manifest.json')}")


if __name__ == "__main__":
    main()
