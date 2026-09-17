"""Shared, dependency-light primitives for the M³FD Thermal-VQA pipeline.

M³FD is deliberately kept separate from the SUN RGB-D release machinery: its
boxes are detection boxes, its two image paths are RGB and thermal (never
``depth_path``), and its capture-group provenance is mandatory.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CANONICAL_CLASSES = ("person", "car", "bus", "motorcycle", "truck", "lamp")
SYNONYMS = {
    "person": "person", "pedestrian": "person", "human": "person",
    "car": "car", "automobile": "car", "vehicle": "car",
    "bus": "bus", "motorcycle": "motorcycle", "motorbike": "motorcycle", "motor bike": "motorcycle",
    "truck": "truck", "lamp": "lamp", "street lamp": "lamp", "streetlight": "lamp",
}


def relative_stem(path: Path, root: Path) -> str:
    """A pairing key that retains subdirectories and only removes the suffix."""
    return path.relative_to(root).with_suffix("").as_posix()


def image_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    result = {relative_stem(path, root): path for path in files}
    if len(result) != len(files):
        raise ValueError(f"duplicate image pairing keys under {root}")
    return result


def annotation_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"annotation directory does not exist: {root}")
    files = [p for p in root.rglob("*.xml") if p.is_file()]
    result = {relative_stem(path, root): path for path in files}
    if len(result) != len(files):
        raise ValueError(f"duplicate annotation pairing keys under {root}")
    return result


def paired_keys(rgb: dict[str, Path], thermal: dict[str, Path], annotations: dict[str, Path]) -> tuple[list[str], dict]:
    """Return complete triplets and an explicit, serialisable mismatch report."""
    rgb_keys, thermal_keys, annotation_keys = set(rgb), set(thermal), set(annotations)
    complete = sorted(rgb_keys & thermal_keys & annotation_keys)
    report = {
        "complete_triplets": len(complete),
        "rgb_without_thermal": sorted(rgb_keys - thermal_keys),
        "rgb_without_annotation": sorted(rgb_keys - annotation_keys),
        "thermal_without_rgb": sorted(thermal_keys - rgb_keys),
        "annotation_without_rgb": sorted(annotation_keys - rgb_keys),
    }
    return complete, report


def canonicalize_class(raw_name: str) -> str | None:
    normalized = " ".join(raw_name.strip().lower().replace("_", " ").replace("-", " ").split())
    return SYNONYMS.get(normalized)


def _number(element: ET.Element | None, field: str, annotation_path: Path) -> float:
    if element is None or element.text is None:
        raise ValueError(f"missing {field} in {annotation_path}")
    value = float(element.text.strip())
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in {annotation_path}")
    return value


def parse_voc_xml(annotation_path: Path) -> tuple[tuple[int, int] | None, list[dict], list[dict]]:
    """Parse Pascal-VOC style M³FD annotations without silently repairing them.

    Returns ``(declared_size, valid_objects, rejected_objects)``. Invalid or
    unknown boxes are reason-coded for the build log rather than becoming gold
    evidence. Coordinates retain their original frame until the indexer maps
    them into the thermal frame.
    """
    try:
        root = ET.parse(annotation_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"unparseable XML {annotation_path}: {error}") from error

    size_node = root.find("size")
    declared_size = None
    if size_node is not None:
        width = int(_number(size_node.find("width"), "size/width", annotation_path))
        height = int(_number(size_node.find("height"), "size/height", annotation_path))
        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive declared annotation size in {annotation_path}")
        declared_size = (width, height)

    valid, rejected = [], []
    for object_index, node in enumerate(root.findall("object")):
        raw_name = (node.findtext("name") or "").strip()
        concept = canonicalize_class(raw_name)
        if concept is None:
            rejected.append({"object_index": object_index, "raw_name": raw_name,
                             "reason_code": "UNKNOWN_CLASS"})
            continue
        try:
            bndbox = node.find("bndbox")
            if bndbox is None:
                raise ValueError("missing bndbox")
            x1 = _number(bndbox.find("xmin"), "xmin", annotation_path)
            y1 = _number(bndbox.find("ymin"), "ymin", annotation_path)
            x2 = _number(bndbox.find("xmax"), "xmax", annotation_path)
            y2 = _number(bndbox.find("ymax"), "ymax", annotation_path)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("non-positive box extent")
        except ValueError as error:
            rejected.append({"object_index": object_index, "raw_name": raw_name,
                             "reason_code": "INVALID_BOX", "detail": str(error)})
            continue
        valid.append({"object_index": object_index, "raw_name": raw_name, "concept": concept,
                      "box_xyxy": [x1, y1, x2, y2]})
    return declared_size, valid, rejected


def clip_box(box_xyxy: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box_xyxy
    clipped = [max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
               max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))]
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def map_box_between_frames(box_xyxy: list[float], source_size: tuple[int, int],
                           target_size: tuple[int, int]) -> list[float]:
    """Scale an axis-aligned box between registered frames of different sizes."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("all mapping dimensions must be positive")
    scale_x, scale_y = target_width / source_width, target_height / source_height
    x1, y1, x2, y2 = box_xyxy
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def capture_group(image_key: str, pattern: str) -> str | None:
    """Extract a defensible grouping key; no implicit frame-random fallback."""
    match = re.search(pattern, image_key)
    if not match:
        return None
    if "group" in match.groupdict() and match.group("group"):
        return match.group("group")
    if match.lastindex:
        return match.group(1)
    return match.group(0)
