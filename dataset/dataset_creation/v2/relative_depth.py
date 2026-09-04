"""
P2 generator for the `relative_depth` question type (plan §4.4) — the
type built specifically to be immune to language priors, since both
candidate objects are named in the question and the correct answer
depends only on measured depth.

Gates: both objects must be the scene's only eligible instance of their
concept (so "the chair" is unambiguous), both must have valid depth, and
the depth gap must clear max(0.3m, 15% of the closer depth) so the
question is not a coin flip on depth-sensor noise.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator_common import load_templates, render_question, run_generator  # noqa: E402
from scene_objects import true_instance_counts  # noqa: E402

TEMPLATES = load_templates("relative_depth.txt")
MIN_GAP_ABSOLUTE_M = 0.3
MIN_GAP_RELATIVE = 0.15


def _clears_gap(depth_a: float, depth_b: float) -> bool:
    gap = abs(depth_a - depth_b)
    return gap >= max(MIN_GAP_ABSOLUTE_M, MIN_GAP_RELATIVE * min(depth_a, depth_b))


def generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger):
    eligible = [obj for obj in resolved_objects if obj["eligible"] and obj["depth_median_m"] is not None]
    concept_counts = true_instance_counts(resolved_objects)
    single_instance = [obj for obj in eligible if concept_counts[obj["concept"]] == 1]

    if len(single_instance) < 2:
        drop_logger.log(scene["image_id"], "FEWER_THAN_TWO_SINGLE_INSTANCE_OBJECTS")
        return []

    candidates = []
    for obj_1, obj_2 in combinations(single_instance, 2):
        if not _clears_gap(obj_1["depth_median_m"], obj_2["depth_median_m"]):
            continue

        closer, farther = (obj_1, obj_2) if obj_1["depth_median_m"] < obj_2["depth_median_m"] else (obj_2, obj_1)
        mention_a, mention_b = (obj_1, obj_2) if rng.random() < 0.5 else (obj_2, obj_1)
        comparative, answer_object = ("closer", closer) if rng.random() < 0.5 else ("farther", farther)

        preposition = "to" if comparative == "closer" else "from"
        template_id, question = render_question(
            TEMPLATES, rng, comparative=comparative, preposition=preposition,
            a=mention_a["display_name"].replace("_", " "),
            b=mention_b["display_name"].replace("_", " "),
        )
        answer = answer_object["display_name"].replace("_", " ")
        candidates.append({
            "variant": comparative, "template_id": template_id, "question": question,
            "answer": answer, "answer_type": "choice",
            "answer_space": f"{mention_a['display_name']}|{mention_b['display_name']}",
            "evidence": {
                "a_concept": obj_1["concept"], "a_depth_m": obj_1["depth_median_m"],
                "b_concept": obj_2["concept"], "b_depth_m": obj_2["depth_median_m"],
                "comparative": comparative, "answer_concept": answer_object["concept"],
            },
        })

    if not candidates:
        drop_logger.log(scene["image_id"], "NO_PAIR_CLEARS_DEPTH_GAP")
    return candidates


if __name__ == "__main__":
    run_generator("relative_depth", generate_candidates_for_scene, seed_offset=4)
