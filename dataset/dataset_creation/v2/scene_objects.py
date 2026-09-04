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


def load_concept_typical_area(path: str) -> dict:
    """Median visible area_frac per concept, written by build_vocab.py (P1).

    Read from a frozen file rather than recomputed here on purpose: a live
    recompute would make one scene's crop verdict depend on which *other*
    scenes are in the corpus, so dropping unrelated train images would
    silently re-gate test items — the same coupling bug Rule S2b hit and
    §13.15 fixed for the RNG and the co-occurrence table.
    """
    with open(path, "r") as typical_area_file:
        return json.load(typical_area_file)["median_area_frac_by_concept"]


def is_cropped_sliver(concept: str, area_frac: float, touches_border: bool,
                       typical_area_by_concept: dict, crop_area_ratio: float) -> bool:
    """True if only a fragment of this instance is inside the frame.

    Requires both signals: touching the border alone is far too common to
    mean anything (measured: 60-89% of released items reference an object
    whose polygon reaches an edge — rooms are photographed from inside, so
    furniture routinely runs off-frame while staying perfectly
    identifiable), and small area alone just means a small object. It is
    the *combination* — reaches the edge, and is much smaller than this
    same concept normally is when fully visible — that says "cut off".

    A concept with too few fully-visible instances to have a typical size
    (P1's MIN_SAMPLES_FOR_TYPICAL_AREA) is never flagged: with no
    reference size there is no evidence of truncation, and guessing would
    drop valid items.
    """
    if not touches_border or area_frac is None:
        return False
    typical_area_frac = typical_area_by_concept.get(concept)
    if not typical_area_frac:
        return False
    return area_frac < crop_area_ratio * typical_area_frac


def resolve_scene_objects(scene: dict, synonym_map: dict, canonical_vocab: dict,
                           min_area_frac: float, typical_area_by_concept: dict | None = None,
                           crop_area_ratio: float = 0.0) -> list:
    """
    Returns one dict per annotated object in the scene:
      object_index, raw_name, concept, display_name, category, is_structural,
      in_vocab, area_frac, centroid_x, centroid_y, depth_median_m,
      depth_valid_frac, is_valid_polygon, touches_border, is_cropped_sliver,
      eligible, reference_eligible

    `eligible` = valid polygon, in canonical vocab, non-structural,
    area_frac >= min_area_frac, and not a frame-cropped sliver. This is the
    set every generator draws *answer* candidates from (Rule V3: a
    structural object is never an answer). `reference_eligible` drops the
    non-structural requirement — it is the set left_right.py may use for
    the non-answer reference object B, since Rule V3 explicitly allows a
    structural object there ("to the left of the door"). Objects that are
    not eligible at all are still returned (raw presence still matters for
    existence-question negatives, Rule V4).

    The crop gate is off when `typical_area_by_concept` is omitted, so a
    caller that only needs presence/geometry need not load P1's table.
    """
    typical_area_by_concept = typical_area_by_concept or {}
    resolved = []
    for obj in scene["objects"]:
        canon = canonicalize(obj["raw_name"], synonym_map, canonical_vocab)
        is_valid_polygon = obj["is_valid_polygon"]
        area_frac = obj["area_frac"]
        touches_border = obj.get("touches_border", False)
        cropped_sliver = is_cropped_sliver(
            canon["concept"], area_frac, touches_border, typical_area_by_concept, crop_area_ratio
        )
        reference_eligible = (
            is_valid_polygon
            and canon["in_vocab"]
            and area_frac is not None
            and area_frac >= min_area_frac
            and not cropped_sliver
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
            "touches_border": touches_border,
            "is_cropped_sliver": cropped_sliver,
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
    area/structural/crop filters that gate `eligible`/`reference_eligible`.
    "Single-instance" gates (relative_depth, nearest_object, left_right)
    must check against this, not against `eligible`: a second instance
    that fails the area threshold, or is half out of frame, is still a
    second chair a human would see in the room, so "the chair" would
    still be ambiguous.
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
