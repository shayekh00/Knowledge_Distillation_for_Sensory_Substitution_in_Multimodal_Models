"""Paired cluster bootstrap over scene groups (experiment_protocol.md §5).

Implements exactly the uncertainty method §5 predeclares before any test result
was seen: resample scene/room groups (`sequence_id`) with replacement, every
question in a sampled group travels together, and the **full five-type macro
metric is recomputed inside each draw** for both models being compared — never
averaged from per-question or per-type intervals. The reported quantity is a
confidence interval on the paired difference (model A macro - model B macro),
not two separate per-model intervals (§5's "Reported quantity" row).

Per-item correctness is computed with the exact same primitives `evaluate.py`
uses (`answers_agree`, `is_missing`) so this can never silently disagree with
the numbers already recorded in each run's `metrics.json`.

Usage::

    python evaluation/paired_bootstrap.py \\
        --predictions-a runs/<D7 run>/predictions.csv --label-a D7 \\
        --predictions-b runs/<B3 run>/predictions.csv --label-b B3 \\
        --split test --json runs/kd/confirmatory_recording/D7_vs_B3_test_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate import (  # noqa: E402
    RELEASE_DIR, VOCAB_DIR, is_missing, load_predictions, load_release_split)
from answer_form import answers_agree  # noqa: E402
from vocab import load_canonical_vocab, load_synonyms  # noqa: E402

N_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260905  # §5: "bootstrap seed 20260905"
QUESTION_TYPES = ("existence", "identify_superlative", "left_right",
                  "nearest_object", "relative_depth")


def per_item_correct(gold: pd.DataFrame, predictions_csv: str, synonym_map: dict,
                     canonical_vocab: dict) -> pd.Series:
    """One boolean per gold row: exact-match-correct under the primary endpoint's
    unconstrained scoring (§2). Mirrors `evaluate.score_predictions`'s inner
    correctness check precisely — an absent prediction counts as wrong."""
    predictions = load_predictions(predictions_csv)
    predicted_by_id = dict(zip(predictions["question_id"], predictions["prediction"]))
    correct = []
    for _, row in gold.iterrows():
        prediction = predicted_by_id.get(row["question_id"])
        has_prediction = not is_missing(prediction)
        is_correct = has_prediction and answers_agree(
            prediction, row["answer"], row["question_type"], synonym_map, canonical_vocab)
        correct.append(is_correct)
    return pd.Series(correct, index=gold.index, dtype=bool)


def cluster_count_arrays(gold: pd.DataFrame, correct_a: pd.Series, correct_b: pd.Series):
    """Per (scene group, question type) totals, as dense arrays indexed
    [group, type] so a bootstrap draw is a single vectorized gather+sum."""
    frame = pd.DataFrame({
        "sequence_id": gold["sequence_id"].values,
        "question_type": gold["question_type"].values,
        "correct_a": correct_a.values,
        "correct_b": correct_b.values,
    })
    groups = frame["sequence_id"].unique()
    group_index = {group: i for i, group in enumerate(groups)}
    n_groups = len(groups)
    n_types = len(QUESTION_TYPES)
    type_index = {qtype: i for i, qtype in enumerate(QUESTION_TYPES)}

    n = np.zeros((n_groups, n_types), dtype=np.int64)
    ca = np.zeros((n_groups, n_types), dtype=np.int64)
    cb = np.zeros((n_groups, n_types), dtype=np.int64)
    agg = frame.groupby(["sequence_id", "question_type"], sort=False).agg(
        n=("correct_a", "size"), ca=("correct_a", "sum"), cb=("correct_b", "sum"))
    for (group, qtype), row in agg.iterrows():
        gi, ti = group_index[group], type_index[qtype]
        n[gi, ti] = row["n"]
        ca[gi, ti] = row["ca"]
        cb[gi, ti] = row["cb"]
    return n, ca, cb, n_groups


def macro_from_counts(n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """n, c shape (..., n_types) -> macro accuracy shape (...). A type with zero
    items in a draw would divide by zero; with 3000+ groups and thousands of
    items per type this is not expected in practice, so it is asserted against
    rather than silently guarded."""
    assert (n > 0).all(), "a bootstrap draw produced a question type with zero items"
    return (c / n).mean(axis=-1)


def bootstrap(n: np.ndarray, ca: np.ndarray, cb: np.ndarray, n_groups: int,
             n_replicates: int, seed: int, chunk: int = 500) -> np.ndarray:
    """Returns the array of `n_replicates` paired differences (macro_a - macro_b),
    one full macro recomputation per draw, chunked to bound peak memory."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_replicates, dtype=np.float64)
    done = 0
    while done < n_replicates:
        take = min(chunk, n_replicates - done)
        idx = rng.integers(0, n_groups, size=(take, n_groups))
        n_draw = n[idx].sum(axis=1)
        ca_draw = ca[idx].sum(axis=1)
        cb_draw = cb[idx].sum(axis=1)
        macro_a = macro_from_counts(n_draw, ca_draw)
        macro_b = macro_from_counts(n_draw, cb_draw)
        diffs[done:done + take] = macro_a - macro_b
        done += take
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-a", required=True)
    parser.add_argument("--predictions-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--release-dir", default=RELEASE_DIR)
    parser.add_argument("--canonical-objects-dir", default=VOCAB_DIR)
    parser.add_argument("--n-replicates", type=int, default=N_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    synonym_map = load_synonyms(os.path.join(VOCAB_DIR, "synonyms.csv"))
    canonical_vocab = load_canonical_vocab(
        os.path.join(args.canonical_objects_dir, "canonical_objects.csv"))
    gold = load_release_split(args.split, args.release_dir).reset_index(drop=True)

    correct_a = per_item_correct(gold, args.predictions_a, synonym_map, canonical_vocab)
    correct_b = per_item_correct(gold, args.predictions_b, synonym_map, canonical_vocab)

    n, ca, cb, n_groups = cluster_count_arrays(gold, correct_a, correct_b)
    macro_a_observed = macro_from_counts(n.sum(axis=0), ca.sum(axis=0))
    macro_b_observed = macro_from_counts(n.sum(axis=0), cb.sum(axis=0))
    diff_observed = float(macro_a_observed - macro_b_observed)

    diffs = bootstrap(n, ca, cb, n_groups, args.n_replicates, args.bootstrap_seed)
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    excludes_zero = ci_low > 0 or ci_high < 0
    meets_threshold = diff_observed >= 0.02 and excludes_zero  # §5.1: >=2pp and CI excludes 0

    result = {
        "label_a": args.label_a, "label_b": args.label_b, "split": args.split,
        "n_test_items": int(len(gold)), "n_scene_groups": int(n_groups),
        "macro_a": float(macro_a_observed), "macro_b": float(macro_b_observed),
        "diff_a_minus_b": diff_observed,
        "n_replicates": args.n_replicates, "bootstrap_seed": args.bootstrap_seed,
        "ci_95_low": ci_low, "ci_95_high": ci_high,
        "ci_excludes_zero": excludes_zero,
        "meets_5_1_threshold": meets_threshold,
        "fraction_replicates_a_not_better": float((diffs <= 0).mean()),
    }
    print(json.dumps(result, indent=2))
    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
