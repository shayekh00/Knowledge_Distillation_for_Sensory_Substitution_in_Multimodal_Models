"""
P2 generator for the `left_right` question type (plan §4.6).

Only left/right is kept (v1's above/under was 2-D-image-plane nonsense —
"the table is under the window" reads oddly even when pixel-true). B, the
reference object, is allowed to be structural (Rule V3: "to the left of
the door" is fine; a structural object is only barred from being an
*answer*), so this generator draws A from the answer-eligible pool and B
from the wider reference-eligible pool.

Overlap is checked with real polygon IoU, not bounding boxes: P0's index
stores only area/centroid per object, so the polygons are rebuilt here
from the raw annotation JSON (same union-and-repair approach as
build_index.py's build_polygon_records, kept local to this one generator
rather than growing the shared P0 schema for a single consumer).
"""
from __future__ import annotations

import json
import os
import sys

from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator_common import load_templates, render_question, run_generator  # noqa: E402
from scene_objects import DATASET_DIR, true_instance_counts  # noqa: E402

TEMPLATES = load_templates("left_right.txt")
MIN_HORIZONTAL_GAP_FRACTION = 0.10
MAX_IOU = 0.20

# Rule V3 allows a structural object as the reference B ("to the left of
# the door"), but wall/floor/ceiling span most of the frame, so their
# centroid sits near the image center regardless of what is actually
# nearby — "left/right of the ceiling" is well-defined but not a
# meaningful spatial question. Localized structural elements (door,
# door_frame, window, ...) are still allowed.
ROOM_SPANNING_REFERENCE_CONCEPTS = {"wall", "floor", "ceiling"}


def _polygons_by_object_index(annotation_absolute_path: str) -> dict:
    with open(annotation_absolute_path, "r") as annotation_file:
        annotation_data = json.load(annotation_file)

    frames = annotation_data.get("frames", [])
    polygon_entries_by_index = {}
    if frames:
        for entry in frames[0].get("polygon", []):
            polygon_entries_by_index.setdefault(entry.get("object"), []).append(entry)

    polygons = {}
    for object_index, entries in polygon_entries_by_index.items():
        parts = []
        for entry in entries:
            xs, ys = entry.get("x", []), entry.get("y", [])
            xs = xs if isinstance(xs, list) else [xs]
            ys = ys if isinstance(ys, list) else [ys]
            if len(xs) < 3:
                continue
            candidate = Polygon(zip(xs, ys)).buffer(0)
            if not candidate.is_empty and candidate.area > 0:
                parts.append(candidate)
        if not parts:
            continue
        union = parts[0]
        for part in parts[1:]:
            union = union.union(part)
        polygons[object_index] = union
    return polygons


def _iou(polygon_a: Polygon, polygon_b: Polygon) -> float:
    intersection = polygon_a.intersection(polygon_b).area
    if intersection == 0:
        return 0.0
    union = polygon_a.union(polygon_b).area
    return intersection / union if union > 0 else 0.0


def passes_pair_gates(object_a: dict, object_b: dict, image_width: float, iou: float) -> bool:
    horizontal_gap = abs(object_a["centroid_x"] - object_b["centroid_x"])
    if horizontal_gap < MIN_HORIZONTAL_GAP_FRACTION * image_width:
        return False
    return iou <= MAX_IOU


def generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger):
    answer_pool = [obj for obj in resolved_objects if obj["eligible"]]
    reference_pool = [
        obj for obj in resolved_objects
        if obj["reference_eligible"] and obj["concept"] not in ROOM_SPANNING_REFERENCE_CONCEPTS
    ]
    concept_counts = true_instance_counts(resolved_objects)

    single_instance_answers = [obj for obj in answer_pool if concept_counts[obj["concept"]] == 1]
    single_instance_references = [obj for obj in reference_pool if concept_counts[obj["concept"]] == 1]

    if not single_instance_answers or len(single_instance_references) < 2:
        drop_logger.log(scene["image_id"], "INSUFFICIENT_SINGLE_INSTANCE_OBJECTS")
        return []

    annotation_absolute_path = os.path.join(DATASET_DIR, scene["annotation_path"])
    polygons_by_index = _polygons_by_object_index(annotation_absolute_path)
    image_width = scene["image_width"]

    candidates = []
    seen_pairs = set()
    for object_a in single_instance_answers:
        for object_b in single_instance_references:
            if object_b["object_index"] == object_a["object_index"]:
                continue
            pair_key = tuple(sorted((object_a["object_index"], object_b["object_index"])))
            if pair_key in seen_pairs:
                continue

            polygon_a = polygons_by_index.get(object_a["object_index"])
            polygon_b = polygons_by_index.get(object_b["object_index"])
            if polygon_a is None or polygon_b is None:
                drop_logger.log(scene["image_id"], "POLYGON_UNAVAILABLE",
                                 f"{object_a['concept']}/{object_b['concept']}")
                continue
            if not passes_pair_gates(object_a, object_b, image_width, _iou(polygon_a, polygon_b)):
                continue

            horizontal_gap = abs(object_a["centroid_x"] - object_b["centroid_x"])
            seen_pairs.add(pair_key)
            answer = "left" if object_a["centroid_x"] < object_b["centroid_x"] else "right"
            template_id, question = render_question(
                TEMPLATES, rng,
                a=object_a["display_name"].replace("_", " "),
                b=object_b["display_name"].replace("_", " "),
            )
            candidates.append({
                "variant": "", "template_id": template_id, "question": question,
                "answer": answer, "answer_type": "choice", "answer_space": "left|right",
                "evidence": {
                    "a_concept": object_a["concept"], "b_concept": object_b["concept"],
                    "a_centroid_x": object_a["centroid_x"], "b_centroid_x": object_b["centroid_x"],
                    "horizontal_gap_px": horizontal_gap, "image_width": image_width,
                },
            })

    if not candidates:
        drop_logger.log(scene["image_id"], "NO_PAIR_CLEARS_GATES")
    return candidates


if __name__ == "__main__":
    run_generator("left_right", generate_candidates_for_scene, seed_offset=6)
