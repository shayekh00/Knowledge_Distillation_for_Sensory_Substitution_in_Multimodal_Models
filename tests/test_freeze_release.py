"""Canonical hashing of the P0 scene index (dataset_creation/v2/freeze_release.py).

`--verify` cried wolf on 2026-09-06 after an environment migration rebuilt
`scene_index.jsonl` with a different numpy/scipy stack: same P0 drop decisions
(11,235/11,235 matched the frozen release exactly), same canonical manifest
hash, but different raw bytes, because per-object geometry/depth floats carry
noise in their last bits. `_sha256_of_manifest` already solves this exact shape
of problem for `manifest.json`'s `built_at_utc`; these tests pin the same
treatment for `scene_index.jsonl`.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "dataset", "dataset_creation", "v2"))

from freeze_release import _round_floats, _sha256_of_scene_index  # noqa: E402


_counter = 0


def write_index(tmp_path, records):
    global _counter
    _counter += 1
    path = str(tmp_path / f"scene_index_{_counter}.jsonl")
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_float_precision_noise_does_not_change_the_hash(tmp_path):
    """The whole point: two runs differing only in float noise must match."""
    a = write_index(tmp_path, [{"image_id": "x", "objects": [{"area_frac": 0.11081933316102352}]}])
    b = write_index(tmp_path, [{"image_id": "x", "objects": [{"area_frac": 0.11081933316102341}]}])
    assert _sha256_of_scene_index(a) == _sha256_of_scene_index(b)


def test_a_real_content_change_still_changes_the_hash(tmp_path):
    """Canonicalising must not become a way to hide a genuine regression."""
    a = write_index(tmp_path, [{"image_id": "x", "objects": [{"raw_name": "chair"}]}])
    b = write_index(tmp_path, [{"image_id": "x", "objects": [{"raw_name": "table"}]}])
    assert _sha256_of_scene_index(a) != _sha256_of_scene_index(b)


def test_a_difference_past_the_rounding_precision_still_changes_the_hash(tmp_path):
    """Rounding must not swallow a difference large enough to matter."""
    a = write_index(tmp_path, [{"image_id": "x", "objects": [{"area_frac": 0.100000}]}])
    b = write_index(tmp_path, [{"image_id": "x", "objects": [{"area_frac": 0.100005}]}])
    assert _sha256_of_scene_index(a) != _sha256_of_scene_index(b)


def test_record_order_still_changes_the_hash(tmp_path):
    """Order is real content here — build_index.py's iteration is deterministic
    with no set()/glob/multiprocessing in the path, so a reordering is a
    genuine change to investigate, not noise to canonicalise away."""
    a = write_index(tmp_path, [{"image_id": "x"}, {"image_id": "y"}])
    b = write_index(tmp_path, [{"image_id": "y"}, {"image_id": "x"}])
    assert _sha256_of_scene_index(a) != _sha256_of_scene_index(b)


def test_key_order_within_a_record_does_not_change_the_hash(tmp_path):
    a = write_index(tmp_path, [{"image_id": "x", "scene_type": "office"}])
    b = write_index(tmp_path, [{"scene_type": "office", "image_id": "x"}])
    assert _sha256_of_scene_index(a) == _sha256_of_scene_index(b)


def test_round_floats_recurses_through_nested_structures():
    value = {"a": [0.123456789, {"b": 0.987654321}], "c": "unchanged", "d": 7}
    rounded = _round_floats(value)
    assert rounded == {"a": [0.123457, {"b": 0.987654}], "c": "unchanged", "d": 7}
