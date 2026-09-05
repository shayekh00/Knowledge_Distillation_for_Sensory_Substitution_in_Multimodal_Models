"""Builds the release-facing artefacts that sit beside the frozen CSVs (§11).

Everything here is *derived* from the frozen release and the build log — it
never changes a released row, so it can be re-run at any time without touching
the freeze. Kept separate from `build_release.py` for exactly that reason: P3
assembles the dataset and must stay reproducible byte-for-byte, while these
files are documentation of what P3 produced.

Writes `manifest.json`, `stats/report.md`, `stats/answer_histograms.png`, and a
consolidated `stats/drops.csv`.

Usage::

    python dataset/dataset_creation/v2/build_release_artifacts.py --version v2.4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # no display in CI or over ssh
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RELEASE_DIR = os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2")
RULE_BASED_DIR = os.path.join(RELEASE_DIR, "rule_based")
STATS_DIR = os.path.join(RELEASE_DIR, "stats")
BUILD_LOG_DIR = os.path.join(PROJECT_ROOT, "build_log")
SPLITS = ("train", "val", "test")
LICENSE_NAME = "CC BY-SA 4.0"


def load_splits() -> dict:
    return {split: pd.read_csv(os.path.join(RULE_BASED_DIR, f"{split}.csv")) for split in SPLITS}


def build_manifest(frames: dict, version: str) -> dict:
    frozen_path = os.path.join(RELEASE_DIR, f"FROZEN_{version}.json")
    with open(frozen_path, encoding="utf-8") as handle:
        frozen = json.load(handle)
    return {
        "name": "VQA-SUNRGBD-v2",
        "version": version,
        "license": LICENSE_NAME,
        "source_dataset": "SUN RGB-D (Song et al., CVPR 2015)",
        "frozen_at_utc": frozen["frozen_at_utc"],
        "freeze_manifest": os.path.basename(frozen_path),
        "question_types": sorted(frames["test"]["question_type"].unique().tolist()),
        "splits": {
            split: {
                "items": int(len(frame)),
                "images": int(frame["image_id"].nunique()),
                "per_question_type": {
                    str(question_type): int(count) for question_type, count
                    in frame["question_type"].value_counts().sort_index().items()},
                "per_sensor": {
                    str(sensor): int(count) for sensor, count
                    in frame["sensor"].value_counts().sort_index().items()},
            } for split, frame in frames.items()},
        "columns": frames["test"].columns.tolist(),
        "evaluation": {
            "script": "evaluate.py",
            "prompt_suffix": "Answer with a single word or number.",
            "decoding": {"temperature": 0, "do_sample": False, "max_new_tokens": 16},
            "headline_metric": "macro accuracy over question types",
            "mandatory_baselines": ["random", "majority", "question_only"],
        },
        "known_limitations": [
            "existence negatives are matched on canonical name only, so a scene "
            "annotated `desk`/`counter`/`coffeetable` can carry a gold `no` for "
            "`table` (defect D17; ~902 items release-wide, see DATASET_CREATION_PLAN.md §13.22)",
            "single-reviewer gold verification only; no inter-rater or kappa claim (§8.3)",
            "some items reference an object largely outside the frame; no filter for "
            "them could be validated (defect D16, §13.17)",
            "re-deriving the vocabulary from scratch yields 148 concepts rather than "
            "the shipped 151; use the committed file (§13.18)",
        ],
    }


def consolidate_drop_log() -> pd.DataFrame:
    """One drops table across P0 and every P2 generator (§8.2)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(BUILD_LOG_DIR, "*drops.csv"))):
        frame = pd.read_csv(path)
        frame["stage"] = os.path.splitext(os.path.basename(path))[0]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_answer_histograms(frames: dict, output_path: str) -> None:
    question_types = sorted(frames["test"]["question_type"].unique())
    figure, axes = plt.subplots(1, len(question_types), figsize=(4 * len(question_types), 3.6))
    for axis, question_type in zip(axes, question_types):
        counts = Counter(frames["test"].loc[
            frames["test"]["question_type"] == question_type, "answer"])
        top = counts.most_common(10)
        axis.barh([answer for answer, _ in reversed(top)],
                  [count for _, count in reversed(top)], color="#4c72b0")
        axis.set_title(f"{question_type}\n(top {len(top)} of {len(counts)})", fontsize=9)
        axis.tick_params(labelsize=7)
    figure.suptitle("VQA-SUNRGBD-v2 — test-split answer distribution", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_stats_report(frames: dict, manifest: dict, drops: pd.DataFrame) -> str:
    lines = [f"# VQA-SUNRGBD-v2 {manifest['version']} — release statistics", "",
             f"Frozen {manifest['frozen_at_utc']} · license {LICENSE_NAME} · "
             f"source: {manifest['source_dataset']}", "",
             "## Size", "",
             "| Split | Items | Images | " + " | ".join(manifest["question_types"]) + " |",
             "|---|---:|---:|" + "---:|" * len(manifest["question_types"])]
    for split in SPLITS:
        info = manifest["splits"][split]
        counts = " | ".join(str(info["per_question_type"].get(question_type, 0))
                            for question_type in manifest["question_types"])
        lines.append(f"| {split} | {info['items']:,} | {info['images']:,} | {counts} |")

    lines += ["", "## Answer balance (test)", "",
              "| Type | Distinct answers | Majority answer | Majority share |",
              "|---|---:|---|---:|"]
    test = frames["test"]
    for question_type in manifest["question_types"]:
        answers = test.loc[test["question_type"] == question_type, "answer"]
        shares = answers.value_counts(normalize=True)
        lines.append(f"| {question_type} | {answers.nunique()} | `{shares.index[0]}` "
                     f"| {shares.iloc[0]:.1%} |")

    lines += ["", "## Sensor composition", "",
              "| Split | " + " | ".join(sorted(manifest["splits"]["test"]["per_sensor"])) + " |",
              "|---|" + "---:|" * len(manifest["splits"]["test"]["per_sensor"])]
    sensors = sorted(manifest["splits"]["test"]["per_sensor"])
    for split in SPLITS:
        per_sensor = manifest["splits"][split]["per_sensor"]
        lines.append(f"| {split} | "
                     + " | ".join(str(per_sensor.get(sensor, 0)) for sensor in sensors) + " |")

    if not drops.empty:
        lines += ["", "## Drop log (§8.2)", "",
                  "Every candidate rejected by a gate, by reason code.", "",
                  "| Reason | Count |", "|---|---:|"]
        for reason, count in drops["reason_code"].value_counts().items():
            lines.append(f"| `{reason}` | {count:,} |")
        lines.append(f"\nFull table: `stats/drops.csv` ({len(drops):,} rows).")

    lines += ["", "## Known limitations", ""]
    lines += [f"* {limitation}" for limitation in manifest["known_limitations"]]
    lines += ["", "Baselines and the evaluation protocol: `evaluate.py` (§9). "
              "Gold verification: `audit/results/report.md` (§8.3).", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v2.4")
    args = parser.parse_args()

    os.makedirs(STATS_DIR, exist_ok=True)
    frames = load_splits()
    manifest = build_manifest(frames, args.version)
    drops = consolidate_drop_log()

    with open(os.path.join(RELEASE_DIR, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if not drops.empty:
        drops.to_csv(os.path.join(STATS_DIR, "drops.csv"), index=False)
    plot_answer_histograms(frames, os.path.join(STATS_DIR, "answer_histograms.png"))
    with open(os.path.join(STATS_DIR, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(render_stats_report(frames, manifest, drops))

    print(f"Wrote manifest.json, stats/report.md, stats/answer_histograms.png"
          f"{', stats/drops.csv' if not drops.empty else ''} for {args.version}.")


if __name__ == "__main__":
    main()
