"""Assign M³FD capture groups to deterministic 70/15/15 splits.

No fallback to individual-frame assignment exists. The command additionally
checks image IDs, capture groups, and optional perceptual near-duplicate groups
for split crossings before it writes anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deterministic_split(records: list[dict], seed: int = 42) -> dict[str, str]:
    """Greedily fill 70/15/15 frame targets in a stable hash order by group."""
    groups = defaultdict(list)
    for record in records:
        group = record.get("_split_group") or record.get("capture_group") or record.get("sequence_id")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{record.get('image_id')}: missing verified capture_group")
        groups[group].append(record)
    total = len(records)
    targets = {"train": total * .70, "val": total * .15, "test": total * .15}
    current = Counter()
    result = {}
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    for group in ordered:
        size = len(groups[group])
        # Pick the split furthest below target; stable tie-break order is part of the contract.
        split = min(("train", "val", "test"), key=lambda name: ((current[name] - targets[name]) / max(targets[name], 1), name))
        result[group] = split
        current[split] += size
    return result


def assert_split_isolation(records: list[dict]) -> None:
    by_image, by_group, by_duplicate = defaultdict(set), defaultdict(set), defaultdict(set)
    for record in records:
        split = record.get("split")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"{record.get('image_id')}: invalid split {split!r}")
        by_image[record["image_id"]].add(split)
        by_group[record.get("capture_group") or record.get("sequence_id")].add(split)
        duplicate_group = record.get("near_duplicate_group")
        if duplicate_group:
            by_duplicate[duplicate_group].add(split)
    crossings = [("image_id", key, value) for key, value in by_image.items() if len(value) > 1]
    crossings += [("capture_group", key, value) for key, value in by_group.items() if len(value) > 1]
    crossings += [("near_duplicate_group", key, value) for key, value in by_duplicate.items() if len(value) > 1]
    if crossings:
        kind, key, splits = crossings[0]
        raise ValueError(f"split leakage: {kind} {key!r} crosses {sorted(splits)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--near-duplicate-groups", type=Path, required=True,
                        help="JSONL emitted by near_duplicates_m3fd.py; mandatory independent leakage check.")
    parser.add_argument("--out", type=Path, default=Path("data/index/scene_index_m3fd.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/index/manifest_m3fd_split.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = load_jsonl(args.index)
    duplicate_rows = load_jsonl(args.near_duplicate_groups)
    duplicate_groups = {row["image_id"]: row["near_duplicate_group"] for row in duplicate_rows}
    if len(duplicate_groups) != len(duplicate_rows) or set(duplicate_groups) != {row["image_id"] for row in records}:
        raise ValueError("near-duplicate assignments must contain each indexed image exactly once")
    for record in records: record["near_duplicate_group"] = duplicate_groups[record["image_id"]]
    # A duplicate may bridge two nominal capture sessions. Treat the connected
    # component as one split unit, not as a post-hoc failure that leaves no
    # usable assignment.
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value: parent[value] = find(parent[value])
        return parent[value]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    for record in records:
        union("capture:" + (record.get("capture_group") or record["sequence_id"]),
              "duplicate:" + record["near_duplicate_group"])
    for record in records:
        record["_split_group"] = find("capture:" + (record.get("capture_group") or record["sequence_id"]))
    assignments = deterministic_split(records, args.seed)
    for record in records:
        record["split"] = assignments[record["_split_group"]]
        del record["_split_group"]
    assert_split_isolation(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    counts = Counter(record["split"] for record in records)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "ratios": {"train": .70, "val": .15, "test": .15},
                   "counts": dict(counts), "capture_groups": len(assignments),
                   "isolation": "passed"}, handle, indent=2, sort_keys=True)
    print(f"assigned {len(records)} frames across {dict(counts)}")


if __name__ == "__main__":
    main()
