"""Draw the single-reviewer gold-verification sample from test.

Implements DATASET_CREATION_PLAN.md §8.3: 150 test items per question_type,
stratified by sensor so each type's audit sample mirrors that type's sensor
mix, drawn with the project's global seed (config default 42).

Usage::

    python -m tools.audit_app.sampling \
        --test-csv release/VQA-SUNRGBD-v2/rule_based/test.csv \
        --out audit/audit_items.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sample_audit_items(items: pd.DataFrame, per_type: int, seed: int) -> pd.DataFrame:
    """Return up to `per_type` rows per `question_type`, stratified by `sensor`.

    Within a type, each sensor's share of the drawn rows matches its share of
    that type's rows as closely as an integer quota allows (largest-remainder
    apportionment). Deterministic for a fixed `seed`; independent of row order.
    """
    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    for _, type_rows in items.groupby("question_type", sort=False):
        target = min(per_type, len(type_rows))
        quotas = _proportional_quotas(type_rows["sensor"].value_counts(), target)
        for sensor, quota in quotas.items():
            if quota == 0:
                continue
            pool = type_rows.index[type_rows["sensor"] == sensor].to_numpy()
            chosen = rng.choice(pool, size=quota, replace=False)
            sampled_parts.append(items.loc[chosen])
    if not sampled_parts:
        return items.iloc[0:0].copy()
    return pd.concat(sampled_parts, ignore_index=True)


def _proportional_quotas(sensor_counts: pd.Series, target: int) -> dict[str, int]:
    """Largest-remainder apportionment of `target` seats across sensors,
    weighted by `sensor_counts`, never exceeding a sensor's own pool size."""
    total = int(sensor_counts.sum())
    if total == 0 or target == 0:
        return {}
    raw_shares = sensor_counts.to_numpy() * target / total
    quotas = np.minimum(np.floor(raw_shares).astype(int), sensor_counts.to_numpy())
    remaining_seats = target - int(quotas.sum())
    if remaining_seats > 0:
        spare_room = sensor_counts.to_numpy() - quotas
        by_largest_remainder = np.argsort(-(raw_shares - np.floor(raw_shares)))
        for idx in by_largest_remainder:
            if remaining_seats <= 0:
                break
            if spare_room[idx] > 0:
                quotas[idx] += 1
                spare_room[idx] -= 1
                remaining_seats -= 1
    return dict(zip(sensor_counts.index, quotas.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, required=True,
                         help="Path to release/VQA-SUNRGBD-v2/rule_based/test.csv")
    parser.add_argument("--out", type=Path, default=Path("audit/audit_items.csv"))
    parser.add_argument("--per-type", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = pd.read_csv(args.test_csv)
    for required_column in ("question_type", "sensor", "question_id"):
        if required_column not in items.columns:
            raise SystemExit(f"--test-csv is missing required column {required_column!r}")

    sampled = sample_audit_items(items, per_type=args.per_type, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.out, index=False)

    print(f"Sampled {len(sampled)} / {len(items)} items -> {args.out}")
    print(sampled.groupby("question_type").size().to_string())


if __name__ == "__main__":
    main()
