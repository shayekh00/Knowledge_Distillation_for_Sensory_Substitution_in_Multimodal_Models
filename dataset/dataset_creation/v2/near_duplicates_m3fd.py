"""Build a conservative RGB-and-thermal dHash near-duplicate grouping for M³FD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def dhash(path: Path) -> int:
    """64-bit difference hash; robust to scale and monotone intensity changes."""
    with Image.open(path) as source:
        pixels = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(pixels.getdata())
    bits = 0
    for y in range(8):
        for x in range(8): bits = (bits << 1) | int(values[y * 9 + x] > values[y * 9 + x + 1])
    return bits


class UnionFind:
    def __init__(self, values): self.parent = {value: value for value in values}
    def find(self, value):
        if self.parent[value] != value: self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def group_hashes(hashes: dict[str, tuple[int, int]], maximum_hamming: int = 4) -> dict[str, str]:
    """Union pairs near in either modality; corpus size is only 4,200 frames."""
    ids = sorted(hashes); groups = UnionFind(ids)
    for offset, image_id in enumerate(ids):
        rgb_hash, thermal_hash = hashes[image_id]
        for candidate in ids[offset + 1:]:
            candidate_rgb, candidate_thermal = hashes[candidate]
            if (rgb_hash ^ candidate_rgb).bit_count() <= maximum_hamming or (thermal_hash ^ candidate_thermal).bit_count() <= maximum_hamming:
                groups.union(image_id, candidate)
    return {image_id: f"nd_{groups.find(image_id)}" for image_id in ids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index/scene_index_m3fd.jsonl"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/index/near_duplicates_m3fd.jsonl"))
    parser.add_argument("--maximum-hamming", type=int, default=4)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line]
    hashes = {row["image_id"]: (dhash(args.dataset_root / row["rgb_path"]), dhash(args.dataset_root / row["thermal_path"])) for row in records}
    groups = group_hashes(hashes, args.maximum_hamming); args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for image_id in sorted(groups): handle.write(json.dumps({"image_id": image_id, "near_duplicate_group": groups[image_id]}) + "\n")
    print(f"wrote {len(groups)} M3FD near-duplicate assignments")


if __name__ == "__main__": main()
