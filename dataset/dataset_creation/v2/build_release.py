"""
P3 of the VQA-SUNRGBD-v2 pipeline (plan §6-§8): assembles the six raw
candidate pools written by P2 (data/candidates/*.csv) into the released,
balanced, stratified per-split CSVs.

Per split:
  1. dedup_one_row_per_image — at most one candidate per (image, type)
  2. val/test only: answer-distribution balancing per type (Rule 6.2) —
     train keeps its natural answer distribution; see balance.py's
     module docstring for why that split is deliberate, not an oversight
  3. val/test only: stratified-by-sensor subsampling to a common item
     count across all six types (Rule 6.4), then a cap of at most 4
     distinct question types per image (Rule 6.4)
  4. assign question_id, write release/VQA-SUNRGBD-v2/rule_based/<split>.csv
  5. run the §8.4 automatic sanity checks and write build_log/p3_report.json
"""
from __future__ import annotations

import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from balance import (  # noqa: E402
    balance_binary,
    cap_distinct_types_per_image,
    cap_majority_share,
    dedup_one_row_per_image,
    dedup_one_row_per_image_targeted,
    scale_to_target_distribution,
    stratified_subsample,
)
from generator_common import CANDIDATE_COLUMNS, CANDIDATES_DIR, DATA_DIR, load_config  # noqa: E402
from scene_objects import DATASET_DIR  # noqa: E402

REPO_ROOT = os.path.dirname(DATA_DIR)
RELEASE_DIR = os.path.join(REPO_ROOT, "release", "VQA-SUNRGBD-v2", "rule_based")
BUILD_LOG_DIR = os.path.join(REPO_ROOT, "build_log")

QUESTION_TYPES = [
    "existence", "count", "identify_superlative", "relative_depth", "nearest_object", "left_right",
]
COUNT_TARGET_DISTRIBUTION = {"1": 0.32, "2": 0.27, "3": 0.19, "4": 0.13, "5": 0.09}
OPEN_VOCAB_MAX_SHARE = 0.08
BINARY_BALANCED_TYPES = {"existence", "left_right"}
TRAIN_TYPE_FLOOR = 0.12
MAX_TYPES_PER_IMAGE_VAL_TEST = 4
RELEASE_COLUMNS = ["question_id"] + CANDIDATE_COLUMNS


def load_candidates(question_type: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(CANDIDATES_DIR, f"{question_type}.csv"), dtype={"answer": str})
    return df[df["split"] == split].reset_index(drop=True)


def balance_for_split(question_type: str, df: pd.DataFrame, split: str, rng: random.Random) -> pd.DataFrame:
    if question_type == "count" and split != "train":
        df = dedup_one_row_per_image_targeted(df, "answer", COUNT_TARGET_DISTRIBUTION, rng)
    else:
        df = dedup_one_row_per_image(df, rng)
    if split == "train" or df.empty:
        return df
    if question_type == "count":
        return scale_to_target_distribution(df, "answer", COUNT_TARGET_DISTRIBUTION, rng)
    if question_type in ("identify_superlative", "nearest_object"):
        return cap_majority_share(df, "answer", OPEN_VOCAB_MAX_SHARE, rng)
    if question_type in BINARY_BALANCED_TYPES:
        return balance_binary(df, "answer", rng)
    return df  # relative_depth: already ~50/50 by construction at generation time


SPLIT_SEED_OFFSETS = {"train": 10, "val": 11, "test": 12}  # fixed, not Python's randomized hash()


def build_split(split: str, config: dict) -> dict:
    rng = random.Random(config["seed"] + SPLIT_SEED_OFFSETS[split])

    per_type_frames = {}
    for question_type in QUESTION_TYPES:
        raw = load_candidates(question_type, split)
        per_type_frames[question_type] = balance_for_split(question_type, raw, split, rng)

    if split in ("val", "test"):
        target_size = min(len(df) for df in per_type_frames.values())
        per_type_frames = {
            qtype: stratified_subsample(df, target_size, "sensor", rng)
            for qtype, df in per_type_frames.items()
        }
        per_type_frames = cap_distinct_types_per_image(per_type_frames, MAX_TYPES_PER_IMAGE_VAL_TEST, rng)
        # The sensor-stratified subsample above only targets Rule 6.3 (sensor
        # mix); it can reintroduce a small amount of answer-balance drift for
        # the exactly-two-class types, so re-tighten those once more here.
        for question_type in BINARY_BALANCED_TYPES:
            per_type_frames[question_type] = balance_binary(per_type_frames[question_type], "answer", rng)

    combined = pd.concat(list(per_type_frames.values()), ignore_index=True)
    combined = combined.sample(frac=1, random_state=rng.randrange(1 << 30)).reset_index(drop=True)
    combined.insert(0, "question_id", [f"{split}_{i:06d}" for i in range(len(combined))])

    os.makedirs(RELEASE_DIR, exist_ok=True)
    output_path = os.path.join(RELEASE_DIR, f"{split}.csv")
    combined[RELEASE_COLUMNS].to_csv(output_path, index=False)

    return {
        "split": split,
        "output_path": output_path,
        "total_items": len(combined),
        "per_type_counts": {qtype: len(df) for qtype, df in per_type_frames.items()},
        "frames": per_type_frames,
    }


def run_sanity_checks(split_results: dict) -> dict:
    checks = {"failures": [], "warnings": []}

    for split, result in split_results.items():
        combined = pd.concat(result["frames"].values(), ignore_index=True)

        duplicate_count = combined.duplicated(subset=["image_id", "question"]).sum()
        if duplicate_count > 0:
            checks["failures"].append(f"{split}: {duplicate_count} duplicate (image_id, question) pairs")

        missing_paths = int(
            (~combined["image_path"].apply(lambda p: os.path.exists(os.path.join(DATASET_DIR, p)))).sum()
        )
        if missing_paths > 0:
            checks["failures"].append(f"{split}: {missing_paths} rows reference a missing image file")

        if split in ("val", "test"):
            type_counts = combined["question_type"].value_counts(normalize=True)
            spread = type_counts.max() - type_counts.min()
            if spread > 0.02:
                checks["warnings"].append(
                    f"{split}: type-balance spread {spread:.1%} exceeds the ±2% target ({type_counts.to_dict()})"
                )

            overall_sensor_mix = combined["sensor"].value_counts(normalize=True)
            for question_type, frame in result["frames"].items():
                if frame.empty:
                    continue
                type_sensor_mix = frame["sensor"].value_counts(normalize=True)
                max_deviation = (type_sensor_mix - overall_sensor_mix).abs().max()
                if pd.isna(max_deviation) or max_deviation > 0.03:
                    checks["warnings"].append(
                        f"{split}/{question_type}: sensor mix deviates {max_deviation:.1%} "
                        f"from the split overall (Rule 6.3 target: <=3%)"
                    )

            for question_type in ("count", "identify_superlative", "nearest_object", "existence", "left_right"):
                frame = result["frames"][question_type]
                if frame.empty:
                    continue
                majority_share = frame["answer"].value_counts(normalize=True).iloc[0]
                cap = {
                    "count": max(COUNT_TARGET_DISTRIBUTION.values()),
                    "identify_superlative": OPEN_VOCAB_MAX_SHARE,
                    "nearest_object": OPEN_VOCAB_MAX_SHARE,
                    "existence": 0.51,
                    "left_right": 0.51,
                }[question_type]
                if majority_share > cap + 0.01:
                    checks["warnings"].append(
                        f"{split}/{question_type}: majority answer share {majority_share:.1%} exceeds target {cap:.1%}"
                    )

        if split == "train":
            type_counts = combined["question_type"].value_counts(normalize=True)
            for question_type, share in type_counts.items():
                if share < TRAIN_TYPE_FLOOR:
                    checks["warnings"].append(
                        f"train/{question_type}: {share:.1%} of train items, below the {TRAIN_TYPE_FLOOR:.0%} floor"
                    )

    return checks


def main() -> None:
    config = load_config()
    split_results = {split: build_split(split, config) for split in ("train", "val", "test")}
    checks = run_sanity_checks(split_results)

    report = {
        "per_split": {
            split: {"total_items": result["total_items"], "per_type_counts": result["per_type_counts"]}
            for split, result in split_results.items()
        },
        "sanity_check_failures": checks["failures"],
        "sanity_check_warnings": checks["warnings"],
    }
    os.makedirs(BUILD_LOG_DIR, exist_ok=True)
    with open(os.path.join(BUILD_LOG_DIR, "p3_report.json"), "w") as report_file:
        json.dump(report, report_file, indent=2)

    print(json.dumps(report, indent=2))
    if checks["failures"]:
        raise SystemExit(f"P3 sanity checks failed: {len(checks['failures'])} failure(s), see build_log/p3_report.json")


if __name__ == "__main__":
    main()
