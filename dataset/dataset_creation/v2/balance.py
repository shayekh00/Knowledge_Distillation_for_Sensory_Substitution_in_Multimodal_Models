"""
Answer- and type-balancing primitives for P3 (plan §6). Every function here
only drops rows — nothing is invented or duplicated to fill a quota, per
Rule 6.2's "seeded drop" method. All randomness takes an explicit
random.Random instance so a rerun with the same seed reproduces the exact
same released rows.

Design decision (recorded in docs/DATASET_CREATION_PLAN.md §6.2): answer-
distribution balancing (target_distribution, majority-share caps) is
applied to val/test only. Train keeps its natural answer distribution
after per-image dedup — the plan's own §6.2 closing paragraph gives the
reasoning for *type* balance ("training-set type imbalance is a modelling
concern, not a benchmark-validity concern") and the same logic extends to
answer balance: a majority-class baseline beating a model is a benchmark-
validity problem only when it happens on the numbers that get reported,
i.e. val/test. Forcing train to match the same target distribution would
be actively harmful here — e.g. `count`'s rarest class (6) only has ~177
train images, and scaling to hit its 7% target would cap the whole
train `count` type at ~2,500 items, under the plan's own 3,000 minimum
(§7) for no benchmark-validity benefit.
"""
from __future__ import annotations

import math
import random

import pandas as pd


def dedup_one_row_per_image(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Exactly one candidate row per image_id, picked uniformly at random.
    Uniform random (not a smarter/greedy pick) is deliberate: any bias in
    which candidate wins would distort the answer distribution we are
    about to measure and balance."""
    kept_positions = []
    for _, group in df.groupby("image_id", sort=False):
        kept_positions.append(rng.choice(list(group.index)))
    return df.loc[kept_positions].reset_index(drop=True)


def dedup_one_row_per_image_targeted(df: pd.DataFrame, answer_column: str,
                                      target_proportions: dict, rng: random.Random) -> pd.DataFrame:
    """
    Like dedup_one_row_per_image, but for a type whose target answer
    distribution is known in advance and heavily skewed (count: most
    images offer several candidate concepts, nearly all answering "1").
    Plain uniform-random dedup discards a scarce answer (e.g. "5")
    whenever it has to compete with a common one in the same image,
    which starves scale_to_target_distribution of real, valid data that
    was available. Here, images are visited in a fixed random order and
    each contributes whichever of its candidates currently has the
    largest deficit against the target proportion — recovering that data
    without inventing or duplicating anything.
    """
    image_ids = list(df["image_id"].unique())
    rng.shuffle(image_ids)
    groups = {image_id: group for image_id, group in df.groupby("image_id")}

    tally = {answer_class: 0 for answer_class in target_proportions}
    kept_positions = []
    for image_id in image_ids:
        group = groups[image_id]
        total_kept = len(kept_positions)
        deficits = {
            index: target_proportions.get(str(row[answer_column]), 0.0)
                   - (tally.get(str(row[answer_column]), 0) / total_kept if total_kept else 0.0)
            for index, row in group.iterrows()
        }
        best_deficit = max(deficits.values())
        best_indices = [index for index, deficit in deficits.items() if deficit == best_deficit]
        chosen = rng.choice(best_indices)
        kept_positions.append(chosen)
        answer_class = str(df.loc[chosen, answer_column])
        tally[answer_class] = tally.get(answer_class, 0) + 1

    return df.loc[kept_positions].reset_index(drop=True)


def scale_to_target_distribution(df: pd.DataFrame, answer_column: str,
                                  target_proportions: dict, rng: random.Random) -> pd.DataFrame:
    """
    Downsample so the answer distribution matches target_proportions
    exactly, retaining as much data as possible: the scale factor is set
    by the class that is scarcest relative to its target share, then
    every other class is trimmed down to match.
    """
    achieved_counts = df[answer_column].astype(str).value_counts().to_dict()
    scale = min(
        achieved_counts.get(answer_class, 0) / proportion
        for answer_class, proportion in target_proportions.items()
        if proportion > 0
    )
    kept_frames = []
    for answer_class, proportion in target_proportions.items():
        target_count = math.floor(scale * proportion)
        class_rows = df[df[answer_column].astype(str) == answer_class]
        if len(class_rows) <= target_count:
            kept_frames.append(class_rows)
        else:
            kept_positions = rng.sample(list(class_rows.index), target_count)
            kept_frames.append(df.loc[kept_positions])
    return pd.concat(kept_frames, ignore_index=True) if kept_frames else df.iloc[0:0]


def cap_majority_share(df: pd.DataFrame, answer_column: str, max_share: float,
                        rng: random.Random, max_iterations: int = 25) -> pd.DataFrame:
    """Iteratively trims any answer class that exceeds max_share of the
    *current* total (the cap is relative to the shrinking pool, so this
    must iterate rather than compute a single pass)."""
    current = df
    for _ in range(max_iterations):
        total = len(current)
        if total == 0:
            return current
        counts = current[answer_column].value_counts()
        max_allowed = math.floor(max_share * total)
        offenders = counts[counts > max_allowed]
        if offenders.empty:
            return current
        kept_frames = []
        for answer_class, group in current.groupby(answer_column):
            if answer_class in offenders.index:
                kept_positions = rng.sample(list(group.index), max_allowed)
                kept_frames.append(current.loc[kept_positions])
            else:
                kept_frames.append(group)
        current = pd.concat(kept_frames, ignore_index=True)
    return current


def balance_binary(df: pd.DataFrame, answer_column: str, rng: random.Random) -> pd.DataFrame:
    """Trims the majority class down to exactly match the minority class
    count, for a two-valued answer column (yes/no, left/right)."""
    counts = df[answer_column].value_counts()
    if len(counts) < 2:
        return df
    minority_count = counts.min()
    kept_frames = []
    for answer_class, group in df.groupby(answer_column):
        if len(group) > minority_count:
            kept_positions = rng.sample(list(group.index), minority_count)
            kept_frames.append(df.loc[kept_positions])
        else:
            kept_frames.append(group)
    return pd.concat(kept_frames, ignore_index=True)


def stratified_subsample(df: pd.DataFrame, target_size: int, stratify_column: str,
                          rng: random.Random) -> pd.DataFrame:
    """
    Subsamples df down to target_size while keeping stratify_column's
    (sensor) proportions as close as possible to df's own current
    proportions — used to bring every type down to the same split-wide
    item count (Rule 6.4) without shifting any one type's sensor mix
    (Rule 6.3).
    """
    if len(df) <= target_size:
        return df
    proportions = df[stratify_column].value_counts(normalize=True)
    quotas = {value: round(proportion * target_size) for value, proportion in proportions.items()}
    # Rounding can miss target_size by a few rows; fix up on the largest bucket.
    drift = target_size - sum(quotas.values())
    if drift != 0:
        largest_bucket = proportions.idxmax()
        quotas[largest_bucket] += drift

    kept_frames = []
    for value, group in df.groupby(stratify_column):
        quota = min(quotas.get(value, 0), len(group))
        if quota > 0:
            kept_positions = rng.sample(list(group.index), quota)
            kept_frames.append(df.loc[kept_positions])
    result = pd.concat(kept_frames, ignore_index=True) if kept_frames else df.iloc[0:0]
    return result


def cap_distinct_types_per_image(per_type_frames: dict, max_types_per_image: int,
                                  rng: random.Random) -> dict:
    """Rule 6.4: a val/test image may contribute at most max_types_per_image
    distinct question types. Drops surplus (image, type) rows at random
    until every image is within the cap."""
    image_to_types = {}
    for question_type, frame in per_type_frames.items():
        for image_id in frame["image_id"]:
            image_to_types.setdefault(image_id, []).append(question_type)

    types_to_drop_per_image = {}
    for image_id, types_present in image_to_types.items():
        if len(types_present) > max_types_per_image:
            surplus = rng.sample(types_present, len(types_present) - max_types_per_image)
            types_to_drop_per_image[image_id] = set(surplus)

    trimmed = {}
    for question_type, frame in per_type_frames.items():
        drop_mask = frame["image_id"].map(
            lambda image_id: question_type in types_to_drop_per_image.get(image_id, set())
        )
        trimmed[question_type] = frame.loc[~drop_mask].reset_index(drop=True)
    return trimmed
