"""Phase 3's train/val/test policy decision for ARKitScenes (arkitscenes_plan.md
§6 Phase 2/3), deferred by `build_index_arkit.py`'s own docstring: that script
labels every record in one `--scans-dir` invocation with a single `--split`
value and does not itself decide train/val/test membership.

The mapping mirrors SUN-RGB-D's own scheme exactly, reusing its tested
`assign_split()` (Rule S2) rather than reimplementing split logic: ARKitScenes'
official Validation fold — the one externally-defined held-out set neither
pipeline invented — plays the same role as SUN-RGB-D's official `allsplit.mat`
test pool, and its official Training fold is the pool `val_fraction` (config's
`split.val_fraction`, 0.15) is carved out of by `GroupShuffleSplit` grouped on
`sequence_id` (here: `video_id`, one per scan — the direct ARKitScenes analogue
of SUN-RGB-D's `sun3d_group_by` room-level grouping, so no two frames from the
same scan ever land in different splits).

Consumes the two raw per-fold indexes `build_index_arkit.py` writes
independently (one `--split train` over the Training-fold scans-dir, one
`--split test` over the Validation-fold scans-dir — those flags are only
placeholders at that stage, overwritten here) and produces one merged
`data/index/scene_index_arkit.jsonl`, matching `build_index.py`'s single-file
convention for the existing SUN-RGB-D index.

Usage::

    python dataset/dataset_creation/v2/assign_split_arkit.py \\
        --train-pool data/index/scene_index_arkit_trainpool_raw.jsonl \\
        --test-pool data/index/scene_index_arkit_testpool_raw.jsonl \\
        --out data/index/scene_index_arkit.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BUILD_LOG_DIR = os.path.join(REPO_ROOT, "build_log")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_index import DROP_REASON, assign_split, load_config  # noqa: E402


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-pool", required=True,
                        help="Raw index over the ARKitScenes Training-fold scans-dir "
                             "(build_index_arkit.py, any --split value — overwritten here).")
    parser.add_argument("--test-pool", required=True,
                        help="Raw index over the ARKitScenes Validation-fold scans-dir "
                             "(build_index_arkit.py, any --split value — overwritten here).")
    parser.add_argument("--config", default=os.path.join(DATA_DIR, "config.yaml"))
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "index", "scene_index_arkit.jsonl"))
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    val_fraction = config["split"]["val_fraction"]

    train_pool_records = load_jsonl(args.train_pool)
    test_pool_records = load_jsonl(args.test_pool)
    all_records = train_pool_records + test_pool_records

    train_pool_ids = {record["image_id"] for record in train_pool_records}
    test_pool_ids = {record["image_id"] for record in test_pool_records}
    assert not (train_pool_ids & test_pool_ids), "train/test pools must not overlap"

    image_ids = [record["image_id"] for record in all_records]
    sequence_ids = [record["sequence_id"] for record in all_records]

    drop_rows: list = []
    split_by_image_id = assign_split(
        image_ids, sequence_ids, train_pool_ids, test_pool_ids, val_fraction, seed, drop_rows)
    # ARKitScenes' Training/Validation fold scans are disjoint by construction
    # (phase2_sample.csv draws each scan into exactly one fold), so Rule S2b's
    # "sequence shared with test" trim — built for SUN-RGB-D's sun3d room
    # groups, where one room can contribute frames to both official pools —
    # can never fire here. Asserted, not just assumed, so a future change to
    # how scans are sampled cannot silently start dropping ARKitScenes rows
    # for a reason that was never meant to apply to it.
    assert not any(row["reason_code"] == DROP_REASON["SEQUENCE_SHARED_WITH_TEST"]
                  for row in drop_rows), "unexpected cross-fold sequence overlap"

    kept_records = []
    for record in all_records:
        split = split_by_image_id.get(record["image_id"])
        if split is None:
            continue  # SPLIT_UNASSIGNED — recorded in drop_rows already
        record["split"] = split
        kept_records.append(record)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(BUILD_LOG_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for record in kept_records:
            handle.write(json.dumps(record) + "\n")

    import pandas as pd
    pd.DataFrame(drop_rows).to_csv(
        os.path.join(BUILD_LOG_DIR, "p1_arkit_split_drops.csv"), index=False)

    from collections import Counter
    split_counts = Counter(record["split"] for record in kept_records)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "dataset/dataset_creation/v2/assign_split_arkit.py",
        "train_pool": os.path.relpath(args.train_pool, REPO_ROOT),
        "test_pool": os.path.relpath(args.test_pool, REPO_ROOT),
        "seed": seed, "val_fraction": val_fraction,
        "counts": {
            "frames_total": len(kept_records),
            "frames_dropped": len(drop_rows),
            "by_split": dict(split_counts),
        },
    }
    manifest_path = os.path.join(os.path.dirname(args.out), "manifest_arkit_split.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"{len(kept_records)} frame records -> {args.out}  {dict(split_counts)}")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
