"""
P2 generator for the `identify_superlative` question type (plan §4.3).

Replaces v1's unverifiable "most prominent object" heuristic with three
superlatives, each gated by a margin so the answer is unambiguous even to
a human glancing at the image:
  - largest:         argmax area_frac; margin: winner >= 1.3x runner-up
  - closest_camera:  argmin depth;     margin: winner <= runner-up / 1.2,
                                                and winner >= 0.4 m
  - farthest_camera: argmax depth;     margin: winner >= 1.2x runner-up

All three are computed independently per scene; a scene can contribute
zero, one, two, or three candidates depending on which margins hold.
Per-answer capping (no class > 8% of a split) is P3's job (plan §6.2).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator_common import answer_appears_in_question, load_templates, render_question, run_generator  # noqa: E402

TEMPLATES_BY_VARIANT = {
    "largest": load_templates("identify_superlative_largest.txt"),
    "closest_camera": load_templates("identify_superlative_closest_camera.txt"),
    "farthest_camera": load_templates("identify_superlative_farthest_camera.txt"),
}

LARGEST_MARGIN = 1.3
DEPTH_MARGIN = 1.2
MIN_CLOSEST_DEPTH_M = 0.4


def _largest_candidate(eligible):
    ranked = sorted(eligible, key=lambda obj: obj["area_frac"], reverse=True)
    if len(ranked) == 1:
        return ranked[0], None
    winner, runner_up = ranked[0], ranked[1]
    if winner["area_frac"] >= LARGEST_MARGIN * runner_up["area_frac"]:
        return winner, runner_up
    return None, runner_up


def _closest_candidate(with_depth):
    ranked = sorted(with_depth, key=lambda obj: obj["depth_median_m"])
    if not ranked:
        return None, None
    if len(ranked) == 1:
        return (ranked[0], None) if ranked[0]["depth_median_m"] >= MIN_CLOSEST_DEPTH_M else (None, None)
    winner, runner_up = ranked[0], ranked[1]
    if winner["depth_median_m"] < MIN_CLOSEST_DEPTH_M:
        return None, runner_up
    if runner_up["depth_median_m"] >= DEPTH_MARGIN * winner["depth_median_m"]:
        return winner, runner_up
    return None, runner_up


def _farthest_candidate(with_depth):
    ranked = sorted(with_depth, key=lambda obj: obj["depth_median_m"], reverse=True)
    if len(ranked) < 2:
        return (ranked[0], None) if ranked else (None, None)
    winner, runner_up = ranked[0], ranked[1]
    if winner["depth_median_m"] >= DEPTH_MARGIN * runner_up["depth_median_m"]:
        return winner, runner_up
    return None, runner_up


def generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger):
    eligible = [obj for obj in resolved_objects if obj["eligible"]]
    with_depth = [obj for obj in eligible if obj["depth_median_m"] is not None]

    candidates = []

    for variant, winner_fn, pool in (
        ("largest", _largest_candidate, eligible),
        ("closest_camera", _closest_candidate, with_depth),
        ("farthest_camera", _farthest_candidate, with_depth),
    ):
        if not pool:
            drop_logger.log(scene["image_id"], "NO_ELIGIBLE_OBJECTS", variant)
            continue
        winner, runner_up = winner_fn(pool)
        if winner is None:
            drop_logger.log(scene["image_id"], "MARGIN_FAIL", variant)
            continue

        display_name = winner["display_name"].replace("_", " ")
        template_id, question = render_question(TEMPLATES_BY_VARIANT[variant], rng)
        answer = display_name
        if answer_appears_in_question(question, answer):
            drop_logger.log(scene["image_id"], "ANSWER_IN_QUESTION", variant)
            continue
        candidates.append({
            "variant": variant, "template_id": template_id, "question": question,
            "answer": answer, "answer_type": "object", "answer_space": "",
            "evidence": {
                "winner_concept": winner["concept"], "winner_object_index": winner["object_index"],
                "winner_area_frac": winner["area_frac"], "winner_depth_m": winner["depth_median_m"],
                "runner_up_concept": runner_up["concept"] if runner_up else None,
                "runner_up_area_frac": runner_up["area_frac"] if runner_up else None,
                "runner_up_depth_m": runner_up["depth_median_m"] if runner_up else None,
            },
        })

    return candidates


if __name__ == "__main__":
    run_generator("identify_superlative", generate_candidates_for_scene, seed_offset=3)
