"""assign_split_arkit.py's train/val/test policy decision (arkitscenes_plan.md
§6 Phase 2/3): the Validation-fold pool must map wholesale to "test", the
Training-fold pool must be carved into train/val by `val_fraction`, no record
may be silently dropped when the two input pools are disjoint (as ARKitScenes'
own fold assignment guarantees), and every scan's frames must land in the same
split as each other (Rule S2's grouping, exercised here via a scan with more
than one sampled frame).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SCRIPT_PATH = os.path.join(PARENT_DIR, "assign_split_arkit.py")


def make_record(image_id: str, sequence_id: str) -> dict:
    return {"image_id": image_id, "sequence_id": sequence_id, "split": "placeholder",
            "objects": [{"raw_name": "chair"}]}


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def run_assign_split(tmp_path, train_records, test_records, val_fraction=0.2, seed=42):
    train_path = os.path.join(tmp_path, "trainpool_raw.jsonl")
    test_path = os.path.join(tmp_path, "testpool_raw.jsonl")
    out_path = os.path.join(tmp_path, "scene_index_arkit.jsonl")
    config_path = os.path.join(tmp_path, "config.yaml")
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump({"seed": seed, "split": {"val_fraction": val_fraction}}, handle)

    result = subprocess.run(
        [sys.executable, SCRIPT_PATH,
         "--train-pool", train_path, "--test-pool", test_path,
         "--config", config_path, "--out", out_path],
        capture_output=True, text=True, cwd=PARENT_DIR)
    assert result.returncode == 0, result.stderr

    with open(out_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_validation_fold_pool_maps_wholesale_to_test(tmp_path):
    # At least two distinct scan groups in the train pool: GroupShuffleSplit
    # cannot carve a val subset out of a single group.
    train_records = ([make_record(f"scan_a_{i}", "scan_a") for i in range(5)]
                     + [make_record(f"scan_c_{i}", "scan_c") for i in range(5)])
    test_records = [make_record(f"scan_b_{i}", "scan_b") for i in range(3)]

    records = run_assign_split(str(tmp_path), train_records, test_records)

    by_id = {record["image_id"]: record["split"] for record in records}
    assert all(by_id[f"scan_b_{i}"] == "test" for i in range(3))


def test_training_fold_pool_is_carved_into_train_and_val_only(tmp_path):
    # Many small single-frame scans so val_fraction has room to carve a
    # non-trivial val subset out of the train pool via GroupShuffleSplit.
    train_records = [make_record(f"scan_{i}_0", f"scan_{i}") for i in range(20)]
    test_records = [make_record("scan_held_0", "scan_held")]

    records = run_assign_split(str(tmp_path), train_records, test_records, val_fraction=0.3)

    train_pool_splits = {record["split"] for record in records
                         if record["sequence_id"] != "scan_held"}
    assert train_pool_splits <= {"train", "val"}
    assert "val" in train_pool_splits  # some rows actually landed in val


def test_no_record_is_dropped_when_pools_are_disjoint(tmp_path):
    train_records = [make_record(f"scan_{i}_0", f"scan_{i}") for i in range(10)]
    test_records = [make_record(f"scan_held_{i}_0", f"scan_held_{i}") for i in range(4)]

    records = run_assign_split(str(tmp_path), train_records, test_records, val_fraction=0.2)

    assert len(records) == len(train_records) + len(test_records)


def test_all_frames_of_one_scan_stay_in_the_same_split():
    """Rule S2's grouping (GroupShuffleSplit on sequence_id): a scan that
    contributes several sampled frames to the train pool must never have some
    of its own frames land in train and others in val."""
    train_records = ([make_record(f"scan_x_{i}", "scan_x") for i in range(4)]
                     + [make_record(f"scan_y_{i}", "scan_y") for i in range(4)])
    test_records = [make_record("scan_held_0", "scan_held")]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        records = run_assign_split(tmp_path, train_records, test_records, val_fraction=0.4)

    splits_by_sequence: dict = {}
    for record in records:
        if record["sequence_id"] in ("scan_x", "scan_y"):
            splits_by_sequence.setdefault(record["sequence_id"], set()).add(record["split"])

    assert all(len(splits) == 1 for splits in splits_by_sequence.values())
