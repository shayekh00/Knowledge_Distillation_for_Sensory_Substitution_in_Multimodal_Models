"""Deterministic, annotation-grounded M³FD Thermal-VQA candidate generation."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from m3fd_common import CANONICAL_CLASSES

TEMPLATES = {
    "existence": ["Is there a {object}?", "Can you see a {object}?", "Does this image contain a {object}?", "Is a {object} visible?", "Is there any {object} here?", "Can a {object} be found in this scene?"],
    "count": ["How many {object}s are visible?", "What is the number of {object}s?", "Count the visible {object}s.", "How many {object}s can you see?", "Tell me the count of {object}s.", "What number of {object}s is present?"],
    "left_right": ["Is the {a} to the left or right of the {b}?", "Where is the {a} relative to the {b}: left or right?", "Is the {a} left or right of the {b}?", "Choose left or right: the {a} is ___ the {b}.", "Relative to the {b}, is the {a} left or right?", "Does the {a} appear left or right of the {b}?"],
    "identify_superlative": ["What is the largest annotated object?", "Which object has the largest visible area?", "Identify the largest visible object.", "What object is biggest in the image?", "Which annotated object occupies the most area?", "Name the object with the largest box."],
}
COLUMNS = ["image_id", "sequence_id", "split", "question_type", "template_id", "question", "answer", "answer_type", "answer_space", "rgb_path", "thermal_path", "source", "evidence"]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1]); area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def _question(question_type: str, rng: random.Random, **fields) -> tuple[int, str]:
    template_id = rng.randrange(len(TEMPLATES[question_type]))
    return template_id, TEMPLATES[question_type][template_id].format(**fields)


def generate(scene: dict, question_type: str, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Return candidates and reason-coded drops for one scene/type."""
    rng = random.Random(f"{seed}:{question_type}:{scene['image_id']}")
    objects = scene["objects"]
    counts = Counter(obj["concept"] for obj in objects)
    candidates, drops = [], []
    def add(template_id, question, answer, answer_type, answer_space, evidence):
        candidates.append({"image_id": scene["image_id"], "sequence_id": scene["sequence_id"], "split": scene["split"],
                           "question_type": question_type, "template_id": template_id, "question": question,
                           "answer": str(answer), "answer_type": answer_type, "answer_space": answer_space,
                           "rgb_path": scene["rgb_path"], "thermal_path": scene["thermal_path"], "source": "rule",
                           "evidence": json.dumps(evidence, sort_keys=True)})
    if question_type == "existence":
        if not scene.get("annotation_complete"):
            return [], [{"image_id": scene["image_id"], "question_type": question_type, "reason_code": "ANNOTATION_COMPLETENESS_UNVERIFIED"}]
        for concept in CANONICAL_CLASSES:
            template_id, question = _question(question_type, rng, object=concept)
            add(template_id, question, "yes" if counts[concept] else "no", "yes_no", "yes|no",
                {"rule": "complete_annotation_presence", "concept": concept, "count": counts[concept],
                 "boxes": [obj["thermal_box_xyxy"] for obj in objects if obj["concept"] == concept]})
    elif question_type == "count":
        if not scene.get("annotation_complete"):
            return [], [{"image_id": scene["image_id"], "question_type": question_type, "reason_code": "ANNOTATION_COMPLETENESS_UNVERIFIED"}]
        for concept, count in sorted(counts.items()):
            if count > 12:
                drops.append({"image_id": scene["image_id"], "question_type": question_type,
                              "reason_code": "COUNT_OUTSIDE_DECLARED_ANSWER_RANGE", "detail": str(count)})
                continue
            template_id, question = _question(question_type, rng, object=concept)
            add(template_id, question, count, "count", "0|1|2|3|4|5|6|7|8|9|10|11|12",
                {"rule": "complete_annotation_count", "concept": concept, "count": count})
    elif question_type == "left_right":
        singles = [obj for obj in objects if counts[obj["concept"]] == 1]
        for index, a in enumerate(singles):
            for b in singles[index + 1:]:
                gap = abs(a["centroid_x"] - b["centroid_x"])
                overlap = iou(a["thermal_box_xyxy"], b["thermal_box_xyxy"])
                if gap < .10 * scene["thermal_width"] or overlap > .20:
                    continue
                template_id, question = _question(question_type, rng, a=a["concept"], b=b["concept"])
                answer = "left" if a["centroid_x"] < b["centroid_x"] else "right"
                add(template_id, question, answer, "choice", "left|right", {"rule": "centroid_gap_and_iou", "a": a["concept"], "b": b["concept"], "a_box": a["thermal_box_xyxy"], "b_box": b["thermal_box_xyxy"], "iou": overlap, "horizontal_gap_px": gap, "minimum_gap_px": .10 * scene["thermal_width"]})
        if not candidates: drops.append({"image_id": scene["image_id"], "question_type": question_type, "reason_code": "NO_PAIR_CLEARS_GATES"})
    elif question_type == "identify_superlative":
        ranked = sorted(objects, key=lambda obj: obj["area_px"], reverse=True)
        if len(ranked) < 2 or ranked[0]["area_px"] < 1.20 * ranked[1]["area_px"]:
            drops.append({"image_id": scene["image_id"], "question_type": question_type, "reason_code": "SUPERLATIVE_MARGIN_FAIL"})
        else:
            template_id, question = _question(question_type, rng)
            add(template_id, question, ranked[0]["concept"], "object", "|".join(CANONICAL_CLASSES), {"rule": "largest_box_area_1.20_margin", "winner": ranked[0]["concept"], "winner_box": ranked[0]["thermal_box_xyxy"], "winner_area_px": ranked[0]["area_px"], "runner_up_box": ranked[1]["thermal_box_xyxy"], "runner_up_area_px": ranked[1]["area_px"]})
    else:
        raise ValueError(f"unknown M3FD question type {question_type}")
    return candidates, drops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index/scene_index_m3fd.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidates_m3fd"))
    parser.add_argument("--log-dir", type=Path, default=Path("build_log/m3fd"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True); args.log_dir.mkdir(parents=True, exist_ok=True)
    scenes = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line]
    if any(scene.get("split") not in {"train", "val", "test"} for scene in scenes): raise SystemExit("index must be split-assigned first")
    for question_type in TEMPLATES:
        rows, drops = [], []
        for scene in sorted(scenes, key=lambda row: row["image_id"]):
            generated, rejected = generate(scene, question_type, args.seed); rows.extend(generated); drops.extend(rejected)
        with (args.out_dir / f"{question_type}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS); writer.writeheader(); writer.writerows(rows)
        with (args.log_dir / f"candidates_{question_type}_drops.jsonl").open("w", encoding="utf-8") as handle:
            for row in drops: handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"{question_type}: {len(rows)} candidates, {len(drops)} drops")


if __name__ == "__main__": main()
