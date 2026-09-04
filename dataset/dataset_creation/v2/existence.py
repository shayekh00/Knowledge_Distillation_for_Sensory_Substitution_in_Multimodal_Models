"""
P2 generator for the `existence` question type (plan §4.1).

For each scene, emits up to two raw candidates — one positive, one hard
negative — and leaves final 50/50 + per-object-parity balancing to P3
(plan §6.4: P2 produces gated candidates, P3 balances/selects). Producing
both polarities per scene, where possible, gives P3 more to balance from.

Hard-negative rule: the sampled absent object must (a) not appear in the
scene under any raw name (Rule V4 — even out-of-vocab/too-small counts as
present), and (b) be plausible: same category as something actually in
the scene, or co-occurring with this scene_type in >=5% of that
scene_type's images corpus-wide. This is what stops the v1 shortcut
"frequent noun => yes" (a random *implausible* negative like "toilet" in
an office is trivially guessable from the noun alone; a plausible one is
not).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator_common import (  # noqa: E402
    load_config,
    load_templates,
    render_question,
    run_generator,
)
from scene_objects import load_scene_index, present_concepts_any  # noqa: E402
from vocab import load_canonical_vocab  # noqa: E402

TEMPLATES = load_templates("existence.txt")
COOCCURRENCE_THRESHOLD = 0.05

_scene_type_presence_fraction = None  # populated once by _build_scene_type_stats()
_all_concepts_by_category = None
_canonical_vocab_cache = None


def _get_canonical_vocab():
    global _canonical_vocab_cache
    if _canonical_vocab_cache is None:
        from generator_common import DATA_DIR
        _canonical_vocab_cache = load_canonical_vocab(os.path.join(DATA_DIR, "vocab", "canonical_objects.csv"))
    return _canonical_vocab_cache


def _build_scene_type_stats():
    """Loads the frozen co-occurrence table written by build_index.py (P0).

    Reading a committed table rather than recomputing over whatever happens
    to be in the scene index is what keeps a scene's hard negative stable:
    recomputing coupled every scene's negative to the exact corpus contents,
    so dropping unrelated train images silently reworded test questions.
    Raw names in the table are mapped through the same Rule V1
    normalise+synonym path used everywhere else.
    """
    global _scene_type_presence_fraction, _all_concepts_by_category
    if _scene_type_presence_fraction is not None:
        return

    import json

    from generator_common import DATA_DIR
    from vocab import load_synonyms, normalize_raw_name

    canonical_vocab = load_canonical_vocab(os.path.join(DATA_DIR, "vocab", "canonical_objects.csv"))
    synonym_map = load_synonyms(os.path.join(DATA_DIR, "vocab", "synonyms.csv"))
    table_path = os.path.join(DATA_DIR, "vocab", "scene_type_cooccurrence.json")
    if not os.path.exists(table_path):
        raise SystemExit(
            f"{table_path} is missing — rerun dataset/dataset_creation/v2/build_index.py (P0) to write it."
        )
    with open(table_path) as table_file:
        table = json.load(table_file)

    _scene_type_presence_fraction = {}
    for scene_type, raw_name_fractions in table["raw_name_fraction"].items():
        by_concept: dict = {}
        for raw_name, fraction in raw_name_fractions.items():
            normalized = normalize_raw_name(raw_name)
            if not normalized:
                continue
            concept = synonym_map.get(normalized, normalized)
            # Several raw names collapse onto one concept; the concept is
            # present if any of them is, so keep the largest fraction.
            by_concept[concept] = max(by_concept.get(concept, 0.0), fraction)
        _scene_type_presence_fraction[scene_type] = by_concept

    _all_concepts_by_category = defaultdict(set)
    for concept, entry in canonical_vocab.items():
        if not entry["is_structural"]:
            _all_concepts_by_category[entry["category"]].add(concept)


def _pick_hard_negative(rng, scene_type, present_concepts, present_categories, canonical_vocab):
    _build_scene_type_stats()
    scene_type_frequent = {
        concept for concept, fraction in _scene_type_presence_fraction.get(scene_type, {}).items()
        if fraction >= COOCCURRENCE_THRESHOLD and concept in canonical_vocab
    }
    category_matched = set()
    for category in present_categories:
        category_matched |= _all_concepts_by_category.get(category, set())

    plausible = (scene_type_frequent | category_matched) - present_concepts
    plausible = {c for c in plausible if not canonical_vocab[c]["is_structural"]}
    if not plausible:
        return None
    return rng.choice(sorted(plausible))


def generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger):
    eligible = [obj for obj in resolved_objects if obj["eligible"]]
    if not eligible:
        drop_logger.log(scene["image_id"], "NO_ELIGIBLE_OBJECTS")
        return []

    candidates = []

    positive_object = rng.choice(eligible)
    template_id, question = render_question(TEMPLATES, rng, object=positive_object["display_name"].replace("_", " "))
    candidates.append({
        "variant": "positive", "template_id": template_id, "question": question,
        "answer": "yes", "answer_type": "yes_no", "answer_space": "yes|no",
        "evidence": {"concept": positive_object["concept"], "object_index": positive_object["object_index"],
                     "area_frac": positive_object["area_frac"]},
    })

    present_concepts = present_concepts_any(resolved_objects)
    present_categories = {obj["category"] for obj in eligible}
    canonical_vocab = _get_canonical_vocab()
    negative_concept = _pick_hard_negative(rng, scene["scene_type"], present_concepts, present_categories, canonical_vocab)

    if negative_concept is None:
        drop_logger.log(scene["image_id"], "NO_HARD_NEGATIVE", f"scene_type={scene['scene_type']}")
    else:
        display_name = canonical_vocab[negative_concept]["display_name"].replace("_", " ")
        template_id, question = render_question(TEMPLATES, rng, object=display_name)
        candidates.append({
            "variant": "negative", "template_id": template_id, "question": question,
            "answer": "no", "answer_type": "yes_no", "answer_space": "yes|no",
            "evidence": {"concept": negative_concept, "reason": "category_or_scene_type_plausible"},
        })

    return candidates


if __name__ == "__main__":
    run_generator("existence", generate_candidates_for_scene, seed_offset=1)
