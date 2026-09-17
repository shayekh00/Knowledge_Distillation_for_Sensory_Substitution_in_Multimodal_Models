from __future__ import annotations

import json
from pathlib import Path

from assign_split_m3fd import assert_split_isolation, deterministic_split
from m3fd_candidates import generate
from near_duplicates_m3fd import group_hashes


def scene():
    return {"image_id": "session_a/frame_1", "sequence_id": "session_a", "capture_group": "session_a", "split": "train",
            "rgb_path": "M3FD/Visible/a.jpg", "thermal_path": "M3FD/Infrared/a.png", "thermal_width": 100, "thermal_height": 100,
            "annotation_complete": True, "objects": [
                {"concept": "car", "centroid_x": 20., "centroid_y": 50., "thermal_box_xyxy": [10., 30., 30., 70.], "area_px": 800.},
                {"concept": "person", "centroid_x": 80., "centroid_y": 50., "thermal_box_xyxy": [70., 35., 90., 65.], "area_px": 600.},
            ]}


def test_candidates_are_deterministic_and_grounded():
    first, _ = generate(scene(), "left_right")
    second, _ = generate(scene(), "left_right")
    assert first == second and first[0]["answer"] == "left"
    assert json.loads(first[0]["evidence"])["horizontal_gap_px"] == 60.
    superlative, drops = generate(scene(), "identify_superlative")
    assert superlative[0]["answer"] == "car" and not drops
    existence, _ = generate(scene(), "existence")
    assert {row["answer"] for row in existence} == {"yes", "no"}


def test_unverified_completeness_blocks_existence_and_count():
    item = scene(); item["annotation_complete"] = False
    for question_type in ("existence", "count"):
        candidates, drops = generate(item, question_type)
        assert candidates == [] and drops[0]["reason_code"] == "ANNOTATION_COMPLETENESS_UNVERIFIED"


def test_capture_groups_stay_together_and_leakage_is_rejected():
    records = [{"image_id": f"a/{i}", "capture_group": "a", "sequence_id": "a"} for i in range(3)] + [{"image_id": "b/0", "capture_group": "b", "sequence_id": "b"}]
    assignment = deterministic_split(records)
    assert assignment["a"] in {"train", "val", "test"}
    for row in records: row["split"] = assignment[row["capture_group"]]
    assert_split_isolation(records)
    records[1]["split"] = "test" if records[0]["split"] != "test" else "train"
    try: assert_split_isolation(records)
    except ValueError as error: assert "capture_group" in str(error)
    else: raise AssertionError("cross-split capture group was accepted")


def test_near_duplicate_groups_join_on_either_modality():
    hashes = {"a": (0b0000, 0b1111), "b": (0b0001, 0b110011), "c": (0b1111, 0b0000)}
    groups = group_hashes(hashes, maximum_hamming=1)
    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]
