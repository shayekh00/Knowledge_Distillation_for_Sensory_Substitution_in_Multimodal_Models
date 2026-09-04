"""
Shared plumbing for the P2 per-type question generators (existence.py,
count.py, identify_superlative.py, relative_depth.py, nearest_object.py,
left_right.py): template loading/rendering, the candidate CSV schema, and
the CLI runner boilerplate every generator would otherwise repeat.

A generator module only needs to implement generate_candidates_for_scene()
(see the docstring in run_generator) — everything else (loading the
config/vocab/index, iterating scenes, writing the CSV and the drop log)
lives here once.
"""
from __future__ import annotations

import csv
import json
import os
import random

import yaml

from scene_objects import (
    load_scene_index,
    load_split_image_ids,
    resolve_scene_objects,
)
from vocab import load_canonical_vocab, load_synonyms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BUILD_LOG_DIR = os.path.join(REPO_ROOT, "build_log")
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

CANDIDATE_COLUMNS = [
    "image_id", "sequence_id", "sensor", "scene_type", "split",
    "question_type", "variant", "template_id", "question", "question_paraphrased",
    "answer", "answer_type", "answer_space", "image_path", "depth_path",
    "source", "evidence",
]


def load_config() -> dict:
    with open(os.path.join(DATA_DIR, "config.yaml"), "r") as config_file:
        return yaml.safe_load(config_file)


def load_templates(filename: str) -> list:
    with open(os.path.join(TEMPLATES_DIR, filename), "r") as template_file:
        templates = [line.rstrip("\n") for line in template_file if line.strip()]
    assert len(templates) >= 6, f"{filename} has fewer than 6 templates (Rule Q1)"
    return templates


def render_question(templates: list, rng: random.Random, **fields) -> tuple:
    template_id = rng.randrange(len(templates))
    return template_id, templates[template_id].format(**fields)


def answer_appears_in_question(question: str, answer: str) -> bool:
    return answer.lower() in question.lower()


class DropLogger:
    def __init__(self, question_type: str):
        self.question_type = question_type
        self.rows = []

    def log(self, image_id: str, reason_code: str, detail: str = ""):
        self.rows.append({"image_id": image_id, "question_type": self.question_type,
                           "reason_code": reason_code, "detail": detail})

    def write(self):
        os.makedirs(BUILD_LOG_DIR, exist_ok=True)
        path = os.path.join(BUILD_LOG_DIR, f"p2_{self.question_type}_drops.csv")
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["image_id", "question_type", "reason_code", "detail"])
            writer.writeheader()
            writer.writerows(self.rows)
        return path


def run_generator(question_type: str, generate_candidates_for_scene, seed_offset: int = 0) -> None:
    """
    generate_candidates_for_scene(scene, resolved_objects, rng, config, drop_logger)
      -> list of dicts, each with at least:
         variant, template_id, question, answer, answer_type, answer_space, evidence (dict)
      Common fields (image_id, sequence_id, sensor, scene_type, split,
      question_type, image_path, depth_path, source) are filled in here.

    Each scene gets its own RNG, seeded from (global seed, question type,
    image_id), rather than sharing one stream across the corpus. That
    matters for reproducibility in a way a shared stream cannot give: a
    scene's questions then depend only on that scene, so changing which
    *other* scenes exist — dropping train images that share a capture
    sequence with test, say — leaves every remaining scene's questions
    byte-identical. With a shared stream, removing one scene shifted the
    draws for every scene after it, silently rewording unrelated test
    questions.

    `seed_offset` keeps each question type on its own stream, so adding or
    removing a type does not perturb the others.
    """
    config = load_config()

    synonym_map = load_synonyms(os.path.join(DATA_DIR, "vocab", "synonyms.csv"))
    canonical_vocab = load_canonical_vocab(os.path.join(DATA_DIR, "vocab", "canonical_objects.csv"))
    scenes = load_scene_index(os.path.join(DATA_DIR, "index", "scene_index.jsonl"))
    scenes.sort(key=lambda scene: scene["image_id"])  # deterministic iteration order

    min_area_frac = config["geometry"]["min_area_frac"]
    drop_logger = DropLogger(question_type)

    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    output_path = os.path.join(CANDIDATES_DIR, f"{question_type}.csv")
    candidate_count = 0

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()

        for scene in scenes:
            resolved_objects = resolve_scene_objects(scene, synonym_map, canonical_vocab, min_area_frac)
            scene_rng = random.Random(f"{config['seed'] + seed_offset}:{question_type}:{scene['image_id']}")
            candidates = generate_candidates_for_scene(scene, resolved_objects, scene_rng, config, drop_logger)
            for candidate in candidates:
                row = {
                    "image_id": scene["image_id"],
                    "sequence_id": scene["sequence_id"],
                    "sensor": scene["sensor"],
                    "scene_type": scene["scene_type"],
                    "split": scene["split"],
                    "question_type": question_type,
                    "variant": candidate.get("variant", ""),
                    "template_id": candidate["template_id"],
                    "question": candidate["question"],
                    "question_paraphrased": "",
                    "answer": candidate["answer"],
                    "answer_type": candidate["answer_type"],
                    "answer_space": candidate["answer_space"],
                    "image_path": scene["rgb_path"],
                    "depth_path": scene["depth_path"],
                    "source": "rule",
                    "evidence": json.dumps(candidate.get("evidence", {})),
                }
                writer.writerow(row)
                candidate_count += 1

    drop_log_path = drop_logger.write()
    print(f"[{question_type}] wrote {candidate_count} candidates -> {output_path}")
    print(f"[{question_type}] {len(drop_logger.rows)} scenes dropped -> {drop_log_path}")
