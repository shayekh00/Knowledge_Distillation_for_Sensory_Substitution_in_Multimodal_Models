"""
Shared per-scene object resolution for the P2 question generators.

Every generator (existence, count, identify_superlative, relative_depth,
nearest_object, left_right) starts from the same resolved object list for
a scene, so gating (Rule "min_area", Rule V3 structural exclusion, Rule
V1 canonicalisation) behaves identically across question types instead of
being re-implemented six times with six chances to disagree.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from vocab import canonicalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")


def load_scene_index(scene_index_path: str) -> list:
    with open(scene_index_path, "r") as index_file:
        return [json.loads(line) for line in index_file]


def load_split_image_ids(splits_dir: str, split_name: str) -> set:
    path = os.path.join(splits_dir, f"{split_name}_images.txt")
    with open(path, "r") as split_file:
        return {line.strip() for line in split_file if line.strip()}


def resolve_scene_objects(scene: dict, synonym_map: dict, canonical_vocab: dict,
                           min_area_frac: float) -> list:
    """
    Returns one dict per annotated object in the scene:
      object_index, raw_name, concept, display_name, category, is_structural,
      in_vocab, area_frac, centroid_x, centroid_y, depth_median_m,
      depth_valid_frac, is_valid_polygon, eligible, reference_eligible

    `eligible` = valid polygon, in canonical vocab, non-structural, and
    area_frac >= min_area_frac. This is the set every generator draws
    *answer* candidates from (Rule V3: a structural object is never an
    answer). `reference_eligible` drops the non-structural requirement —
    it is the set left_right.py may use for the non-answer reference
    object B, since Rule V3 explicitly allows a structural object there
    ("to the left of the door"). Objects that are not eligible at all are
    still returned (raw presence still matters for existence-question
    negatives, Rule V4).
    """
    resolved = []
    for obj in scene["objects"]:
        canon = canonicalize(obj["raw_name"], synonym_map, canonical_vocab)
        is_valid_polygon = obj["is_valid_polygon"]
        area_frac = obj["area_frac"]
        reference_eligible = (
            is_valid_polygon
            and canon["in_vocab"]
            and area_frac is not None
            and area_frac >= min_area_frac
        )
        eligible = reference_eligible and not canon["is_structural"]
        resolved.append({
            "object_index": obj["object_index"],
            "raw_name": obj["raw_name"],
            "concept": canon["concept"],
            "display_name": canon["display_name"],
            "category": canon["category"],
            "is_structural": canon["is_structural"],
            "in_vocab": canon["in_vocab"],
            "is_valid_polygon": is_valid_polygon,
            "area_frac": area_frac,
            "centroid_x": obj["centroid_x"],
            "centroid_y": obj["centroid_y"],
            "depth_median_m": obj["depth_median_m"],
            "depth_valid_frac": obj["depth_valid_frac"],
            "eligible": eligible,
            "reference_eligible": reference_eligible,
        })
    return resolved


def eligible_concept_counts(resolved_objects: list) -> Counter:
    return Counter(obj["concept"] for obj in resolved_objects if obj["eligible"])


def true_instance_counts(resolved_objects: list) -> Counter:
    """
    How many physical instances of each concept actually exist in the
    scene, judged by every valid, in-vocab polygon regardless of the
    area/structural filters that gate `eligible`/`reference_eligible`.
    "Single-instance" gates (relative_depth, nearest_object, left_right)
    must check against this, not against `eligible`: a second, smaller
    instance that fails the area threshold is still a second chair a
    human would see in the room, so "the chair" would still be ambiguous.
    """
    return Counter(
        obj["concept"] for obj in resolved_objects
        if obj["is_valid_polygon"] and obj["in_vocab"]
    )


def present_concepts_any(resolved_objects: list) -> set:
    """Every concept present in the scene under any name, eligible or not
    (Rule V4: an out-of-vocab or too-small object still blocks it from
    being used as a false 'absent' existence negative)."""
    return {obj["concept"] for obj in resolved_objects}


def scene_dir_absolute(image_id: str) -> str:
    return os.path.join(DATASET_DIR, "SUNRGBD", image_id)
