"""Build the additive VQA-M3FD-Thermal-v1 release from M³FD candidates.

This command is intentionally strict: it blocks rather than publishing when
capture/duplicate isolation, required balance, shortcut, or minimum-size gates
cannot be met. It never writes to either existing release directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from assign_split_m3fd import assert_split_isolation, load_jsonl

TYPES = ("existence", "count", "left_right", "identify_superlative")
COLUMNS = ["question_id", "image_id", "sequence_id", "split", "question_type", "template_id", "question", "answer", "answer_type", "answer_space", "source", "rgb_path", "thermal_path", "evidence"]


def _load_candidates(directory: Path, question_type: str) -> list[dict]:
    with (directory / f"{question_type}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one_per_image(rows: list[dict], seed: int) -> list[dict]:
    """Retain a deterministic one-row-per-(image,type) selection for val/test."""
    rng = random.Random(seed); grouped = defaultdict(list)
    for row in rows: grouped[row["image_id"]].append(row)
    return [rng.choice(sorted(group, key=lambda row: (row["question"], row["answer"]))) for _, group in sorted(grouped.items())]


def _balance_binary(rows: list[dict], group_key, seed: int) -> list[dict]:
    rng = random.Random(seed); by_group = defaultdict(lambda: defaultdict(list))
    for row in rows: by_group[group_key(row)][row["answer"]].append(row)
    kept = []
    for key, answers in sorted(by_group.items()):
        n = min(len(answers["yes"]), len(answers["no"])) if {"yes", "no"} <= set(answers) else 0
        if not n: continue
        for answer in ("yes", "no"):
            values = sorted(answers[answer], key=lambda row: row["image_id"]); rng.shuffle(values); kept.extend(values[:n])
    return kept


def _cap_majority(rows: list[dict], maximum: float, seed: int) -> list[dict]:
    """Trim answer groups until the selected majority is at most ``maximum``."""
    rng = random.Random(seed); by_answer = defaultdict(list)
    for row in rows: by_answer[row["answer"]].append(row)
    for values in by_answer.values(): rng.shuffle(values)
    if not by_answer: return []
    # Trimming every class to ``maximum * original_total`` is insufficient:
    # removing the majority shrinks the denominator too. Remove exactly from
    # the current majority until the *selected* distribution satisfies the
    # cap, preserving as many rows as possible under a deterministic shuffle.
    selected = {answer: list(values) for answer, values in by_answer.items()}
    while sum(map(len, selected.values())):
        total = sum(map(len, selected.values()))
        majority = max(selected, key=lambda answer: (len(selected[answer]), answer))
        if len(selected[majority]) / total <= maximum + 1e-12:
            break
        selected[majority].pop()
    return [row for values in selected.values() for row in values]


def _tfidf_baseline(train: list[dict], target: list[dict], question_type: str) -> tuple[float, float]:
    train_rows = [row for row in train if row["question_type"] == question_type]
    target_rows = [row for row in target if row["question_type"] == question_type]
    if not train_rows or not target_rows: raise ValueError(f"{question_type}: missing train or target rows")
    majority = Counter(row["answer"] for row in train_rows).most_common(1)[0][0]
    majority_score = sum(row["answer"] == majority for row in target_rows) / len(target_rows)
    labels = {row["answer"] for row in train_rows}
    if len(labels) < 2: return majority_score, majority_score
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform([row["question"] for row in train_rows])
    model = LogisticRegression(max_iter=1000, random_state=42).fit(matrix, [row["answer"] for row in train_rows])
    prediction = model.predict(vectorizer.transform([row["question"] for row in target_rows]))
    return majority_score, sum(a == b["answer"] for a, b in zip(prediction, target_rows)) / len(target_rows)


def build(index_path: Path, candidates_dir: Path, release_root: Path, seed: int = 42) -> dict:
    index = load_jsonl(index_path); assert_split_isolation(index)
    missing_duplicate_evidence = [row["image_id"] for row in index if not row.get("near_duplicate_group")]
    if missing_duplicate_evidence:
        raise ValueError("near-duplicate groups are required before release (first missing: " + missing_duplicate_evidence[0] + ")")
    by_type = {question_type: _load_candidates(candidates_dir, question_type) for question_type in TYPES}
    selections: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        combined = []
        for question_type, source_rows in by_type.items():
            rows = [dict(row) for row in source_rows if row["split"] == split]
            if split in {"val", "test"}: rows = _one_per_image(rows, seed + len(question_type))
            if question_type == "existence":
                rows = _balance_binary(rows, lambda row: json.loads(row["evidence"])["concept"], seed)
            elif question_type == "left_right":
                # normalize left/right into yes/no to reuse the exact balancer, then restore it.
                for row in rows: row["_binary"] = "yes" if row["answer"] == "left" else "no"
                swapped = [{**row, "answer": row["_binary"]} for row in rows]
                selected = _balance_binary(swapped, lambda _row: "all", seed)
                ids = {(row["image_id"], row["question"]) for row in selected}
                rows = [row for row in rows if (row["image_id"], row["question"]) in ids]
            elif split in {"val", "test"}:
                rows = _cap_majority(rows, .35, seed + len(question_type))
            if not rows: raise ValueError(f"{split}/{question_type}: no rows survive required gates")
            if split in {"val", "test"} and len(rows) < 200:
                raise ValueError(f"{split}/{question_type}: {len(rows)} rows; plan requires at least 200")
            combined.extend(rows)
        selections[split] = combined
    shortcut = {}
    for split in ("val", "test"):
        for question_type in TYPES:
            majority, tfidf = _tfidf_baseline(selections["train"], selections[split], question_type)
            shortcut[f"{split}/{question_type}"] = {"majority": majority, "tfidf": tfidf}
            if tfidf > majority + .05 + 1e-12: raise ValueError(f"{split}/{question_type}: TF-IDF {tfidf:.1%} exceeds majority {majority:.1%} by >5pp")
    rule_based = release_root / "rule_based"; rule_based.mkdir(parents=True, exist_ok=True)
    report = {"counts": {}, "shortcut_baselines": shortcut}
    for split, rows in selections.items():
        rng = random.Random(f"{seed}:release:{split}"); rng.shuffle(rows)
        for offset, row in enumerate(rows): row["question_id"] = f"m3fd_{split}_{offset:07d}"
        with (rule_based / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
        report["counts"][split] = dict(Counter(row["question_type"] for row in rows))
    (release_root / "stats").mkdir(exist_ok=True)
    (release_root / "stats" / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index/scene_index_m3fd.jsonl"))
    parser.add_argument("--candidates-dir", type=Path, default=Path("data/candidates_m3fd"))
    parser.add_argument("--release-root", type=Path, default=Path("release/VQA-M3FD-Thermal-v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); report = build(args.index, args.candidates_dir, args.release_root, args.seed)
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__": main()
