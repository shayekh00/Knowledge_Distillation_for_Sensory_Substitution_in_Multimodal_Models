"""Build the M³FD RGB/thermal scene index after the Phase-0 provenance gate.

The command accepts an extracted source under ``dataset/`` and emits no VQA
release. ``--capture-group-regex`` is required: M³FD frames must never be
split randomly when capture provenance has not been established.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3fd_common import (annotation_files, capture_group, clip_box, image_files,
                         map_box_between_frames, paired_keys, parse_voc_xml)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUT = REPO_ROOT / "data" / "index" / "scene_index_m3fd.jsonl"
DEFAULT_LOG = REPO_ROOT / "build_log" / "m3fd"


def dataset_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(DATASET_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"M³FD assets must be under {DATASET_ROOT}; got {path}") from error


def load_review(path: Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    reviews = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = row.get("image_id")
            valid = row.get("registration_valid")
            if not isinstance(image_id, str) or not isinstance(valid, bool):
                raise ValueError(f"{path}:{line_number}: require image_id and boolean registration_valid")
            if image_id in reviews:
                raise ValueError(f"duplicate registration review for {image_id}")
            reviews[image_id] = valid
    return reviews


def build_record(image_id: str, rgb_path: Path, thermal_path: Path, annotation_path: Path,
                 group_pattern: str, coordinate_frame: str, annotation_complete: bool,
                 review: dict[str, bool], require_review: bool, drops: list[dict]) -> dict | None:
    group = capture_group(image_id, group_pattern)
    if not group:
        drops.append({"image_id": image_id, "reason_code": "CAPTURE_GROUP_UNVERIFIED", "detail": group_pattern})
        return None
    if require_review and review.get(image_id) is not True:
        reason = "REGISTRATION_REJECTED" if image_id in review else "REGISTRATION_UNREVIEWED"
        drops.append({"image_id": image_id, "reason_code": reason, "detail": ""})
        return None
    if review.get(image_id) is False:
        drops.append({"image_id": image_id, "reason_code": "REGISTRATION_REJECTED", "detail": ""})
        return None
    try:
        declared_size, objects, rejected = parse_voc_xml(annotation_path)
    except ValueError as error:
        drops.append({"image_id": image_id, "reason_code": "ANNOTATION_UNPARSEABLE", "detail": str(error)})
        return None
    for rejected_object in rejected:
        drops.append({"image_id": image_id, **rejected_object})
    if not objects:
        drops.append({"image_id": image_id, "reason_code": "NO_VALID_OBJECTS", "detail": ""})
        return None
    with Image.open(rgb_path) as image:
        rgb_size = image.size
    with Image.open(thermal_path) as image:
        thermal_size = image.size
    source_size = thermal_size if coordinate_frame == "thermal" else rgb_size
    if declared_size is not None and declared_size != source_size:
        drops.append({"image_id": image_id, "reason_code": "ANNOTATION_DIMENSION_MISMATCH",
                      "detail": f"declared={declared_size}, expected_{coordinate_frame}={source_size}"})
        return None
    object_records = []
    for obj in objects:
        original = clip_box(obj["box_xyxy"], *source_size)
        if original is None:
            drops.append({"image_id": image_id, "object_index": obj["object_index"],
                          "raw_name": obj["raw_name"], "reason_code": "BOX_OUTSIDE_SOURCE_FRAME", "detail": ""})
            continue
        thermal_box = original if coordinate_frame == "thermal" else map_box_between_frames(original, rgb_size, thermal_size)
        thermal_box = clip_box(thermal_box, *thermal_size)
        if thermal_box is None:
            drops.append({"image_id": image_id, "object_index": obj["object_index"],
                          "raw_name": obj["raw_name"], "reason_code": "BOX_OUTSIDE_THERMAL_FRAME", "detail": ""})
            continue
        x1, y1, x2, y2 = thermal_box
        area = (x2 - x1) * (y2 - y1)
        if area < 1.0:
            drops.append({"image_id": image_id, "object_index": obj["object_index"],
                          "raw_name": obj["raw_name"], "reason_code": "TINY_THERMAL_BOX", "detail": str(area)})
            continue
        object_records.append({
            "object_index": obj["object_index"], "raw_name": obj["raw_name"], "concept": obj["concept"],
            "original_box_xyxy": original, "thermal_box_xyxy": thermal_box,
            "area_px": area, "area_frac": area / float(thermal_size[0] * thermal_size[1]),
            "centroid_x": (x1 + x2) / 2.0, "centroid_y": (y1 + y2) / 2.0,
            "is_valid_polygon": True, "polygon_xy": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        })
    if not object_records:
        drops.append({"image_id": image_id, "reason_code": "NO_USABLE_THERMAL_OBJECTS", "detail": ""})
        return None
    return {
        "image_id": image_id, "sequence_id": group, "capture_group": group, "split": None,
        "sensor": "thermal", "scene_type": "unknown", "image_width": thermal_size[0], "image_height": thermal_size[1],
        "rgb_width": rgb_size[0], "rgb_height": rgb_size[1], "thermal_width": thermal_size[0], "thermal_height": thermal_size[1],
        "rgb_path": dataset_relative(rgb_path), "thermal_path": dataset_relative(thermal_path),
        "annotation_path": dataset_relative(annotation_path), "box_coordinate_frame": coordinate_frame,
        "annotation_complete": annotation_complete, "registration_valid": review.get(image_id),
        "objects": object_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Extracted M³FD directory, under dataset/.")
    parser.add_argument("--rgb-dir", default="Visible")
    parser.add_argument("--thermal-dir", default="Infrared")
    parser.add_argument("--annotation-dir", default="Annotation")
    parser.add_argument("--capture-group-regex", required=True,
                        help="Regex over relative image stem; first/named 'group' capture is the session key.")
    parser.add_argument("--box-coordinate-frame", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--annotation-complete", action="store_true",
                        help="Only set after a source-level completeness finding; enables existence/count later.")
    parser.add_argument("--registration-review", type=Path)
    parser.add_argument("--require-registration-review", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    rgb, thermal, annotations = (image_files(source_root / args.rgb_dir), image_files(source_root / args.thermal_dir),
                                 annotation_files(source_root / args.annotation_dir))
    keys, pairing_report = paired_keys(rgb, thermal, annotations)
    review, drops, records = load_review(args.registration_review), [], []
    for key in keys:
        record = build_record(key, rgb[key], thermal[key], annotations[key], args.capture_group_regex,
                              args.box_coordinate_frame, args.annotation_complete, review,
                              args.require_registration_review, drops)
        if record is not None:
            records.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (args.log_dir / "index_drops.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "object_index", "raw_name", "reason_code", "detail"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(drops)
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(source_root),
                "rgb_dir": args.rgb_dir, "thermal_dir": args.thermal_dir, "annotation_dir": args.annotation_dir,
                "box_coordinate_frame": args.box_coordinate_frame, "capture_group_regex": args.capture_group_regex,
                "annotation_complete": args.annotation_complete, "require_registration_review": args.require_registration_review,
                "pairing": pairing_report, "records_written": len(records), "drop_count": len(drops)}
    with (args.log_dir / "index_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"wrote {len(records)} verified M³FD records to {args.out}")


if __name__ == "__main__":
    main()
