"""Read-only Phase-0 probe for an extracted M³FD source.

It inventories pair/annotation consistency, selects a deterministic
class-stratified review sample, and writes RGB/thermal box overlays. It never
creates an index or release; a human must record the registration verdicts
before ``build_index_m3fd.py --require-registration-review`` is allowed to
keep a record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3fd_common import (annotation_files, image_files, map_box_between_frames,
                         paired_keys, parse_voc_xml)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_sample(keys: list[str], classes_by_key: dict[str, set[str]], size: int, seed: int) -> list[str]:
    """Give every observed class a turn, then fill deterministically from all pairs."""
    if size <= 0:
        return []
    rng = random.Random(seed)
    chosen: list[str] = []
    by_class: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        for concept in classes_by_key[key]:
            by_class[concept].append(key)
    for concept in sorted(by_class):
        candidates = sorted(by_class[concept])
        chosen.append(rng.choice(candidates))
    seen = set(chosen)
    remainder = [key for key in sorted(keys) if key not in seen]
    rng.shuffle(remainder)
    return (chosen + remainder)[:min(size, len(keys))]


def draw_overlay(image_path: Path, boxes: list[list[float]], destination: Path, colour: str) -> None:
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=colour, width=max(1, min(image.size) // 200))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", default="Visible")
    parser.add_argument("--thermal-dir", default="Infrared")
    parser.add_argument("--annotation-dir", default="Annotation")
    parser.add_argument("--box-coordinate-frame", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-terms", required=True, help="URL or local path to the source terms.")
    parser.add_argument("--download", type=Path, help="Optional downloaded archive to checksum.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.source_root.resolve()
    rgb, thermal, annotations = (image_files(root / args.rgb_dir), image_files(root / args.thermal_dir),
                                 annotation_files(root / args.annotation_dir))
    keys, pairing = paired_keys(rgb, thermal, annotations)
    classes_by_key, parse_failures, dimensions = {}, [], Counter()
    valid_keys = []
    for key in keys:
        try:
            declared_size, objects, rejected = parse_voc_xml(annotations[key])
            with Image.open(rgb[key]) as image:
                rgb_size = image.size
            with Image.open(thermal[key]) as image:
                thermal_size = image.size
        except Exception as error:  # retained as probe evidence, never hidden
            parse_failures.append({"image_id": key, "error": str(error)})
            continue
        valid_keys.append(key)
        classes_by_key[key] = {obj["concept"] for obj in objects}
        dimensions[f"rgb={rgb_size},thermal={thermal_size},annotation={declared_size}"] += 1

    sample = stratified_sample(valid_keys, classes_by_key, args.sample_size, args.seed)
    review_rows = []
    for key in sample:
        _, objects, _ = parse_voc_xml(annotations[key])
        with Image.open(rgb[key]) as image:
            rgb_size = image.size
        with Image.open(thermal[key]) as image:
            thermal_size = image.size
        boxes = [obj["box_xyxy"] for obj in objects]
        thermal_boxes = boxes if args.box_coordinate_frame == "thermal" else [
            map_box_between_frames(box, rgb_size, thermal_size) for box in boxes]
        safe_name = key.replace("/", "__")
        draw_overlay(rgb[key], boxes if args.box_coordinate_frame == "rgb" else [
            map_box_between_frames(box, thermal_size, rgb_size) for box in boxes],
                     args.out_dir / "overlays" / f"{safe_name}_rgb.png", "lime")
        draw_overlay(thermal[key], thermal_boxes,
                     args.out_dir / "overlays" / f"{safe_name}_thermal.png", "red")
        review_rows.append({"image_id": key, "rgb_path": str(rgb[key]), "thermal_path": str(thermal[key]),
                            "annotation_path": str(annotations[key]), "registration_valid": None,
                            "reviewer": None, "notes": ""})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "registration_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(), "phase": "M3FD Phase 0 read-only probe",
        "source_url": args.source_url, "license_terms": args.license_terms,
        "download": str(args.download) if args.download else None,
        "download_sha256": sha256(args.download) if args.download else None,
        "source_root": str(root), "layout": {"rgb_dir": args.rgb_dir, "thermal_dir": args.thermal_dir,
                                                  "annotation_dir": args.annotation_dir},
        "box_coordinate_frame_claim": args.box_coordinate_frame, "pairing": pairing,
        "xml_parse_failures": parse_failures, "dimensions": dict(dimensions),
        "classes_by_annotation": dict(Counter(c for classes in classes_by_key.values() for c in classes)),
        "sample_size_requested": args.sample_size, "sample_image_ids": sample,
        "manual_gate": "Fill registration_review.jsonl; at least 95% valid mappings and a verified capture group are required before construction.",
    }
    with (args.out_dir / "probe_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"probed {len(valid_keys)}/{len(keys)} parseable triplets; wrote {len(sample)} review overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
