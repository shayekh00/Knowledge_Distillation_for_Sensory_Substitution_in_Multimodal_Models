"""Freeze or verify VQA-M3FD-Thermal-v1 after all empirical gates pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_RELEASE_FILES = ("rule_based/train.csv", "rule_based/val.csv", "rule_based/test.csv", "DATASHEET.md", "stats/build_report.json")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def freeze(release_root: Path, version: str = "v1.0") -> dict:
    missing = [name for name in REQUIRED_RELEASE_FILES if not (release_root / name).is_file()]
    if missing: raise ValueError("cannot freeze missing files: " + ", ".join(missing))
    hashes = {name: digest(release_root / name) for name in REQUIRED_RELEASE_FILES}
    manifest = {"dataset": "VQA-M3FD-Thermal-v1", "version": version,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "files": hashes}
    (release_root / "FROZEN_v1.0.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify(release_root: Path) -> None:
    path = release_root / "FROZEN_v1.0.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = digest(release_root / name)
        if actual != expected: raise ValueError(f"freeze drift: {name} ({actual} != {expected})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path("release/VQA-M3FD-Thermal-v1"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify: verify(args.release_root); print("M3FD freeze verification passed")
    else: print(json.dumps(freeze(args.release_root), indent=2))


if __name__ == "__main__": main()
