"""
P2 generator for the `nearest_object` question type (plan §4.5).

Distance is computed in metres in the camera's 3-D frame (pinhole
back-projection of each object's 2-D centroid using its median depth and
the scene's intrinsics.txt — see depth_utils.backproject_to_camera_frame
for why skipping the world-alignment rotation Rtilt is exact here, not an
approximation), not in 2-D pixels like v1's proximity/direction
generators. A candidate answer must also belong to a different canonical
concept than the anchor object, so "the closest chair to the chair" can
never be generated.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_utils import backproject_to_camera_frame, load_intrinsics  # noqa: E402
from generator_common import answer_appears_in_question, load_templates, render_question, run_generator  # noqa: E402
from scene_objects import scene_dir_absolute, true_instance_counts  # noqa: E402

TEMPLATES = load_templates("nearest_object.txt")
MARGIN_RATIO = 0.8


def _euclidean_distance(point_a, point_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger):
    eligible = [obj for obj in resolved_objects if obj["eligible"] and obj["depth_median_m"] is not None]
    concept_counts = true_instance_counts(resolved_objects)
    anchors = [obj for obj in eligible if concept_counts[obj["concept"]] == 1]

    if not anchors:
        drop_logger.log(scene["image_id"], "NO_SINGLE_INSTANCE_ANCHOR")
        return []

    camera_intrinsics = load_intrinsics(scene_dir_absolute(scene["image_id"]))
    if camera_intrinsics is None:
        drop_logger.log(scene["image_id"], "MISSING_INTRINSICS")
        return []

    points_3d = {
        obj["object_index"]: backproject_to_camera_frame(
            obj["centroid_x"], obj["centroid_y"], obj["depth_median_m"], camera_intrinsics
        )
        for obj in eligible
    }

    candidates = []
    for anchor in anchors:
        pool = [obj for obj in eligible if obj["object_index"] != anchor["object_index"]
                and obj["concept"] != anchor["concept"]]
        if len(pool) < 2:
            drop_logger.log(scene["image_id"], "INSUFFICIENT_CANDIDATES", anchor["concept"])
            continue

        ranked = sorted(pool, key=lambda obj: _euclidean_distance(points_3d[anchor["object_index"]],
                                                                   points_3d[obj["object_index"]]))
        nearest, second_nearest = ranked[0], ranked[1]
        nearest_distance = _euclidean_distance(points_3d[anchor["object_index"]], points_3d[nearest["object_index"]])
        second_distance = _euclidean_distance(points_3d[anchor["object_index"]], points_3d[second_nearest["object_index"]])
        if second_distance == 0 or nearest_distance > MARGIN_RATIO * second_distance:
            drop_logger.log(scene["image_id"], "MARGIN_FAIL", anchor["concept"])
            continue

        template_id, question = render_question(TEMPLATES, rng, object=anchor["display_name"].replace("_", " "))
        answer = nearest["display_name"].replace("_", " ")
        if answer_appears_in_question(question, answer):
            drop_logger.log(scene["image_id"], "ANSWER_IN_QUESTION", f"{anchor['concept']}/{nearest['concept']}")
            continue
        candidates.append({
            "variant": "", "template_id": template_id, "question": question,
            "answer": answer, "answer_type": "object", "answer_space": "",
            "evidence": {
                "anchor_concept": anchor["concept"], "anchor_object_index": anchor["object_index"],
                "answer_concept": nearest["concept"], "answer_object_index": nearest["object_index"],
                "nearest_distance_m": round(nearest_distance, 3),
                "second_nearest_distance_m": round(second_distance, 3),
            },
        })

    return candidates


if __name__ == "__main__":
    run_generator("nearest_object", generate_candidates_for_scene, seed_offset=5)
