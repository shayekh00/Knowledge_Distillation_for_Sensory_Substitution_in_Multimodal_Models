"""
Freezes the current release/VQA-SUNRGBD-v2/ into a versioned, checksummed
snapshot (plan §8.1: "the builder writes manifest.json with the git commit,
config hash, seed, toolbox checksum, and per-type row counts... Two runs on
two machines must produce byte-identical CSVs").

This is the point past which the released CSVs are locked: any future
change to the pipeline (a generator, a balancing rule, the vocabulary) must
produce a *new* version, not silently rewrite this one. FROZEN.json is the
record that lets that be enforced — `--verify` recomputes every hash and
fails loudly if anything drifted.

Usage::

    python dataset/dataset_creation/v2/freeze_release.py --version v1.0
    python dataset/dataset_creation/v2/freeze_release.py --verify v1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RELEASE_ROOT = os.path.join(REPO_ROOT, "release", "VQA-SUNRGBD-v2")
RULE_BASED_DIR = os.path.join(RELEASE_ROOT, "rule_based")

# Every input that determines the release's content. If any of these
# changes, regenerating would not reproduce this exact release.
TRACKED_INPUTS = [
    os.path.join(DATA_DIR, "config.yaml"),
    os.path.join(DATA_DIR, "vocab", "synonyms.csv"),
    os.path.join(DATA_DIR, "vocab", "canonical_objects.csv"),
    os.path.join(DATA_DIR, "vocab", "scene_type_cooccurrence.json"),
    os.path.join(DATA_DIR, "vocab", "concept_typical_area.json"),
    os.path.join(DATA_DIR, "splits", "train_images.txt"),
    os.path.join(DATA_DIR, "splits", "val_images.txt"),
    os.path.join(DATA_DIR, "splits", "test_images.txt"),
    os.path.join(DATA_DIR, "index", "manifest.json"),
]
RELEASE_FILES = ["train.csv", "val.csv", "test.csv"]


# Fields that change on every run without the build's *content* changing.
# Hashing them makes --verify cry wolf after any rebuild, which trains the
# reader to ignore a drift report — worse than not checking at all.
VOLATILE_MANIFEST_FIELDS = {"built_at_utc"}


def sha256_of(path: str) -> str:
    if os.path.basename(path) == "manifest.json":
        return _sha256_of_manifest(path)
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of_manifest(path: str) -> str:
    """Hash the P0 manifest's meaningful content only.

    The manifest records a wall-clock `built_at_utc`, so its raw bytes
    differ after every P0 run even when the config, toolbox checksums and
    per-type counts it exists to pin are all unchanged. Those substantive
    fields are what a freeze needs to detect drift in; the build time is not.
    """
    with open(path, "r") as manifest_file:
        manifest = json.load(manifest_file)
    substantive = {k: v for k, v in manifest.items() if k not in VOLATILE_MANIFEST_FIELDS}
    canonical = json.dumps(substantive, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def row_count(path: str) -> int:
    with open(path, "r") as csv_file:
        return sum(1 for _ in csv_file) - 1  # minus header


def build_manifest(version: str) -> dict:
    missing = [path for path in TRACKED_INPUTS if not os.path.exists(path)]
    if missing:
        raise SystemExit("Cannot freeze: missing input file(s):\n  " + "\n  ".join(missing))

    release_hashes = {}
    for filename in RELEASE_FILES:
        path = os.path.join(RULE_BASED_DIR, filename)
        if not os.path.exists(path):
            raise SystemExit(f"Cannot freeze: {path} does not exist — run build_release.py first.")
        release_hashes[filename] = {"sha256": sha256_of(path), "rows": row_count(path)}

    input_hashes = {
        os.path.relpath(path, REPO_ROOT): sha256_of(path) for path in TRACKED_INPUTS
    }

    return {
        "version": version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_files": release_hashes,
        "tracked_inputs": input_hashes,
        "note": (
            "Any change to a tracked input or a rule_based/*.csv file invalidates "
            "this version. Bump `version` and re-freeze rather than editing in place. "
            "Verify with --verify."
        ),
    }


def verify(version: str) -> None:
    frozen_path = os.path.join(RELEASE_ROOT, f"FROZEN_{version}.json")
    if not os.path.exists(frozen_path):
        raise SystemExit(f"No frozen manifest at {frozen_path}")
    with open(frozen_path) as frozen_file:
        frozen = json.load(frozen_file)

    problems = []
    for filename, recorded in frozen["release_files"].items():
        path = os.path.join(RULE_BASED_DIR, filename)
        if not os.path.exists(path):
            problems.append(f"{filename}: file is missing")
            continue
        actual = sha256_of(path)
        if actual != recorded["sha256"]:
            problems.append(f"{filename}: sha256 mismatch (frozen {recorded['sha256'][:12]}..., "
                             f"now {actual[:12]}...)")

    for relative_path, recorded_hash in frozen["tracked_inputs"].items():
        path = os.path.join(REPO_ROOT, relative_path)
        if not os.path.exists(path):
            problems.append(f"{relative_path}: file is missing")
            continue
        actual = sha256_of(path)
        if actual != recorded_hash:
            problems.append(f"{relative_path}: sha256 mismatch — an input changed since freezing")

    if problems:
        print(f"DRIFT DETECTED from {version}:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"{version}: verified, no drift. {frozen_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="Freeze the current release under this version tag, e.g. v1.0")
    group.add_argument("--verify", help="Verify the current release still matches this frozen version")
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
        return

    version = args.version
    target_path = os.path.join(RELEASE_ROOT, f"FROZEN_{version}.json")
    if os.path.exists(target_path):
        raise SystemExit(f"{target_path} already exists — versions are immutable once frozen. "
                          f"Use a new version tag.")

    manifest = build_manifest(version)
    with open(target_path, "w") as target_file:
        json.dump(manifest, target_file, indent=2, sort_keys=True)

    print(f"Frozen as {version} -> {target_path}")
    for filename, info in manifest["release_files"].items():
        print(f"  {filename}: {info['rows']:6d} rows  sha256={info['sha256'][:16]}...")
    print(f"\nVerify any time with:\n  python dataset/dataset_creation/v2/freeze_release.py --verify {version}")


if __name__ == "__main__":
    main()
