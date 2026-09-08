"""Phase 2 bulk download driver (arkitscenes_plan.md §5/§6).

Downloads the 750 scans in `phase2_sample.csv` (seed 42, fold-proportional:
668 Training + 82 Validation, matching the full corpus's 89.1%/10.9% split)
via the vendored, zipfile-patched `download_data.py`, one scan at a time.

Resume-safe by construction, matching this project's own established pattern
for a wall-clock-limited server (experiment_protocol.md's 2026-09-08 restart-
safety row): a scan already extracted on disk is skipped rather than
re-fetched, so an interrupted run costs at most the one scan in flight.

Not batched-then-pruned per an earlier draft of §5 — that step existed to
save disk, and disk is no longer the binding constraint (569 GB free,
corrected 2026-09-08); the full download for all 750 scans is estimated at
~750 x 123 MB ~= 92 GB, comfortably inside budget. Simpler to just keep
every frame build_index_arkit.py might want to sample from, rather than
pruning frames it hasn't been asked to sample yet.

Usage::

    python dataset/dataset_creation/arkit_tools/download_phase2.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARKIT_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(REPO_ROOT, "dataset", "ARKitScenes")
SAMPLE_CSV = os.path.join(ARKIT_TOOLS_DIR, "phase2_sample.csv")

sys.path.insert(0, os.path.join(ARKIT_TOOLS_DIR, "ARKitScenes"))


def already_downloaded(fold: str, video_id: str) -> bool:
    # download_data() writes to {download_dir}/3dod/{fold}/{video_id}/... —
    # matches process_one_scan's own path construction in build_index_arkit.py.
    scan_dir = os.path.join(DOWNLOAD_DIR, "3dod", fold, video_id)
    frames_dir = os.path.join(scan_dir, f"{video_id}_frames")
    annotation = os.path.join(scan_dir, f"{video_id}_3dod_annotation.json")
    return os.path.isdir(frames_dir) and os.path.isfile(annotation)


def main() -> None:
    import download_data  # the vendored, zipfile-patched script (module-level side effects only on call)

    with open(SAMPLE_CSV) as handle:
        rows = list(csv.DictReader(handle))

    started = time.time()
    done = skipped = failed = 0
    for index, row in enumerate(rows):
        video_id, fold = row["video_id"], row["fold"]
        if already_downloaded(fold, video_id):
            skipped += 1
            continue
        try:
            download_data.download_data(
                dataset="3dod", video_ids=[video_id], dataset_splits=[fold],
                download_dir=DOWNLOAD_DIR, keep_zip=False,
                raw_dataset_assets=None, should_download_laser_scanner_point_cloud=False)
            done += 1
        except Exception as error:
            failed += 1
            print(f"  FAILED {fold}/{video_id}: {error}", flush=True)
            continue

        if (index + 1) % 10 == 0:
            elapsed_min = (time.time() - started) / 60
            print(f"  {index + 1}/{len(rows)}  done={done} skipped={skipped} failed={failed}  "
                  f"elapsed={elapsed_min:.1f}min", flush=True)

    print(f"PHASE2_DOWNLOAD_COMPLETE: {done} downloaded, {skipped} already present, "
          f"{failed} failed, {len(rows)} total", flush=True)


if __name__ == "__main__":
    main()
