"""
P2 generator for the `count` question type (plan §4.2).

Emits one candidate per (scene, eligible concept) whose eligible instance
count falls in 1..6 — every valid candidate, not a pre-balanced sample.
Hitting the target answer distribution ([0.30, 0.25, 0.18, 0.12, 0.08,
0.07] for 1..6) is P3's job (plan §6.2); P2's only responsibility is
correctness of each candidate: the count is over eligible instances only
(valid polygon, area >= min_area_frac), matching what a human could
actually verify by looking at the image, not the raw, unfiltered
annotation object list (that was v1's Count bug: it counted every object
name occurrence, visible or not).
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import inflect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator_common import answer_appears_in_question, load_templates, render_question, run_generator  # noqa: E402

TEMPLATES = load_templates("count.txt")
_INFLECT = inflect.engine()
# Capped at 5, not 6: a count of exactly 6 identical objects is rare enough
# in the real corpus (long tail) that hitting the target answer
# distribution (balance.py / build_release.py) at an adequate release size
# was not achievable with 6 as an answer class — see
# docs/DATASET_CREATION_PLAN.md §6.2 / §13 for the measured numbers.
MAX_COUNT = 5


def _plural_form(display_name: str) -> str:
    spaced = display_name.replace("_", " ")
    plural = _INFLECT.plural_noun(spaced)
    return plural if plural else spaced + "s"


def generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger):
    eligible = [obj for obj in resolved_objects if obj["eligible"]]
    if not eligible:
        drop_logger.log(scene["image_id"], "NO_ELIGIBLE_OBJECTS")
        return []

    concept_to_display = {obj["concept"]: obj["display_name"] for obj in eligible}
    counts = Counter(obj["concept"] for obj in eligible)
    countable = {concept: count for concept, count in counts.items() if 1 <= count <= MAX_COUNT}

    if not countable:
        drop_logger.log(scene["image_id"], "NO_COUNTABLE_CONCEPT",
                         f"max_count_seen={max(counts.values()) if counts else 0}")
        return []

    candidates = []
    for concept, count in countable.items():
        display_name = concept_to_display[concept]
        template_id, question = render_question(TEMPLATES, rng, object_plural=_plural_form(display_name))
        answer = str(count)
        if answer_appears_in_question(question, answer):
            drop_logger.log(scene["image_id"], "ANSWER_IN_QUESTION", concept)
            continue
        candidates.append({
            "variant": "", "template_id": template_id, "question": question,
            "answer": answer, "answer_type": "number", "answer_space": "1|2|3|4|5",
            "evidence": {"concept": concept, "count": count,
                         "object_indices": [o["object_index"] for o in eligible if o["concept"] == concept]},
        })
    return candidates


if __name__ == "__main__":
    run_generator("count", generate_candidates_for_scene, seed_offset=2)
