"""Draw the single-reviewer gold-verification sample from test.

Implements DATASET_CREATION_PLAN.md §8.3: 150 test items per question_type,
stratified by sensor so each type's audit sample mirrors that type's sensor
mix, drawn with the project's global seed (config default 42).

`--source` fills in that release's test CSV and audit directory from
sources.py, along with what to stratify on. M³FD strata are `None`: its release
rows carry no `sensor` column (`build_release_m3fd.COLUMNS`) and every frame
comes off the same thermal sensor, so its 150-per-type sample — the one
M3FD_THERMAL_VQA_RUNBOOK.md's freeze gate depends on — is drawn uniformly
within each type instead.

Usage::

    python -m tools.audit_app.sampling --source sunrgbd
    python -m tools.audit_app.sampling --source m3fd
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.audit_app.sources import SOURCES, resolve_source


def sample_audit_items(items: pd.DataFrame, per_type: int, seed: int,
                       stratify_by: str | None = "sensor") -> pd.DataFrame:
    """Return up to `per_type` rows per `question_type`.

    With `stratify_by` set, each stratum's share of the drawn rows matches its
    share of that type's rows as closely as an integer quota allows
    (largest-remainder apportionment). With it None, the per-type quota is
    drawn uniformly. Deterministic for a fixed `seed`; independent of row order
    either way.
    """
    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    for _, type_rows in items.groupby("question_type", sort=False):
        target = min(per_type, len(type_rows))
        if target == 0:
            continue
        if stratify_by is None:
            chosen = rng.choice(type_rows.index.to_numpy(), size=target, replace=False)
            sampled_parts.append(items.loc[chosen])
            continue
        quotas = _proportional_quotas(type_rows[stratify_by].value_counts(), target)
        for stratum, quota in quotas.items():
            if quota == 0:
                continue
            pool = type_rows.index[type_rows[stratify_by] == stratum].to_numpy()
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
    parser.add_argument("--source", default="sunrgbd", choices=sorted(SOURCES),
                        help="Which dataset to sample; supplies the defaults below.")
    parser.add_argument("--test-csv", type=Path,
                        help="Release test CSV. Defaults to the source's own.")
    parser.add_argument("--out", type=Path,
                        help="Where to write the sample. Defaults to <source audit dir>/audit_items.csv.")
    parser.add_argument("--per-type", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by",
                        help="Override the source's stratification column.")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Draw each type's quota uniformly, ignoring strata.")
    args = parser.parse_args()

    source = resolve_source(args.source)
    test_csv = args.test_csv or source.release_test_csv_path
    out_path = args.out or (source.audit_dir_path / "audit_items.csv")
    stratify_by = None if args.no_stratify else (args.stratify_by or source.stratify_by)

    if not test_csv.is_file():
        raise SystemExit(
            f"{test_csv} does not exist — build the {source.title} release first.")
    items = pd.read_csv(test_csv)
    required = ["question_type", "question_id"] + ([stratify_by] if stratify_by else [])
    for required_column in required:
        if required_column not in items.columns:
            raise SystemExit(f"{test_csv} is missing required column {required_column!r}")

    sampled = sample_audit_items(items, per_type=args.per_type, seed=args.seed,
                                 stratify_by=stratify_by)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    strata = f"stratified by {stratify_by}" if stratify_by else "unstratified within type"
    print(f"[{source.name}] sampled {len(sampled)} / {len(items)} items ({strata}) -> {out_path}")
    print(sampled.groupby("question_type").size().to_string())


if __name__ == "__main__":
    main()
