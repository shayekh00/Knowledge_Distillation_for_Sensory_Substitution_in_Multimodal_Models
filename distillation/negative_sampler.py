"""Scene-aware negative sampling for the contrastive bank (plan §8.3).

`contrastive_loss` refuses an empty negative bank, but refusing is only half the
requirement: the negatives must also be *valid*. The plan is specific — negatives
are drawn from other training scenes, excluding the scene itself and its known
room/sequence neighbours, and the number of distinct scenes per candidate set is
logged rather than assumed.

Two failure modes this module exists to prevent, both recorded in
`docs/New_Submission/implementation_audit.md` §B4:

* treating multiple crops of one scene as independent negatives, which quietly
  turns a contrastive objective into a near-duplicate detector;
* assuming gradient accumulation enlarges the candidate set. It does not — it
  enlarges the optimizer step. The bank is the only thing that supplies negatives.

Sampling is seeded and deterministic: the same scene, seed, and bank produce the
same negatives on every run, so a contrastive ablation is reproducible.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

DEFAULT_NEGATIVES = 255


@dataclass(frozen=True)
class SceneBank:
    """Training scenes available as contrastive negatives.

    Attributes:
        scene_ids: one entry per bank row, in feature-matrix order.
        sequence_of: scene id -> sequence/room group, used to exclude neighbours.
    """
    scene_ids: list[str]
    sequence_of: dict[str, str]

    def __post_init__(self):
        missing = [scene for scene in self.scene_ids if scene not in self.sequence_of]
        if missing:
            raise ValueError(
                f"{len(missing)} bank scene(s) have no sequence id, so their room "
                f"neighbours cannot be excluded: {missing[:3]}")

    @property
    def n_scenes(self) -> int:
        return len(set(self.scene_ids))

    @property
    def n_sequences(self) -> int:
        return len(set(self.sequence_of[scene] for scene in self.scene_ids))


def build_scene_bank(rows) -> SceneBank:
    """Bank index from release rows. One entry per **distinct** scene.

    Deduplicating by scene is what stops several questions about one image, or
    several crops of it, from appearing as independent negatives.
    """
    sequence_of: dict[str, str] = {}
    order: list[str] = []
    for row in rows:
        scene = str(row["image_id"])
        if scene not in sequence_of:
            sequence_of[scene] = str(row["sequence_id"])
            order.append(scene)
    return SceneBank(scene_ids=order, sequence_of=sequence_of)


def eligible_negative_indices(bank: SceneBank, anchor_scene: str) -> np.ndarray:
    """Bank rows usable as negatives for `anchor_scene`.

    Excludes the anchor itself and every scene sharing its sequence id — frames
    of the same physical room are not independent negatives.
    """
    anchor_sequence = bank.sequence_of.get(anchor_scene)
    return np.array([
        index for index, scene in enumerate(bank.scene_ids)
        if scene != anchor_scene and bank.sequence_of[scene] != anchor_sequence
    ], dtype=np.int64)


def sample_negatives(bank: SceneBank, anchor_scene: str, n_negatives: int = DEFAULT_NEGATIVES,
                     seed: int = 0, allow_fewer: bool = False) -> np.ndarray:
    """Deterministically sample bank indices to use as negatives.

    Args:
        n_negatives: requested count. The plan's starting point is 255 negatives
            plus one positive.
        allow_fewer: when the eligible pool is smaller than requested, return the
            whole pool instead of raising. Off by default — silently training on
            fewer negatives than declared makes two runs incomparable.

    Raises:
        ValueError: if no eligible negative exists, or if the pool is too small
            and `allow_fewer` is not set.
    """
    if n_negatives < 1:
        raise ValueError(f"n_negatives must be at least 1, got {n_negatives}")
    eligible = eligible_negative_indices(bank, anchor_scene)
    if eligible.size == 0:
        raise ValueError(
            f"no eligible negatives for scene {anchor_scene!r}: every bank entry is "
            "the anchor itself or a room/sequence neighbour. A contrastive objective "
            "cannot be formed here (audit B4).")
    if eligible.size < n_negatives:
        if not allow_fewer:
            raise ValueError(
                f"only {eligible.size} eligible negatives for scene {anchor_scene!r}, "
                f"{n_negatives} requested. Pass allow_fewer=True to accept a smaller "
                "bank, and record the actual count — it is part of the configuration.")
        return np.sort(eligible)
    generator = np.random.default_rng([seed, _stable_hash(anchor_scene)])
    return np.sort(generator.choice(eligible, size=n_negatives, replace=False))


def _stable_hash(text: str) -> int:
    """Seed component that does not vary between interpreter runs.

    Python's built-in `hash` is salted per process, so using it here would make
    "deterministic" sampling reproducible only within a single run.
    """
    value = 0
    for character in text:
        value = (value * 131 + ord(character)) % (2 ** 31 - 1)
    return value


def candidate_set_report(bank: SceneBank, anchor_scene: str,
                         sampled: np.ndarray) -> dict:
    """Diagnostics the plan requires logging for every candidate set."""
    scenes = [bank.scene_ids[index] for index in sampled]
    sequences = [bank.sequence_of[scene] for scene in scenes]
    return {
        "anchor_scene": anchor_scene,
        "anchor_sequence": bank.sequence_of.get(anchor_scene),
        "n_negatives": len(sampled),
        "n_distinct_scenes": len(set(scenes)),
        "n_distinct_sequences": len(set(sequences)),
        "bank_scenes": bank.n_scenes,
        "bank_sequences": bank.n_sequences,
    }


def assert_valid_candidate_set(report: dict) -> None:
    """Fail loudly on a candidate set that cannot support the objective."""
    if report["n_negatives"] < 1:
        raise ValueError("contrastive candidate set has no negatives (audit B4)")
    if report["n_distinct_scenes"] != report["n_negatives"]:
        raise ValueError(
            f"candidate set has {report['n_negatives']} negatives but only "
            f"{report['n_distinct_scenes']} distinct scenes — duplicates are not "
            "independent negatives")
    if report["anchor_sequence"] is not None and report["n_distinct_sequences"] < 1:
        raise ValueError("candidate set contains no distinct room groups")


def bank_statistics(bank: SceneBank) -> dict:
    """Bank-level summary for the run manifest."""
    per_sequence = collections.Counter(bank.sequence_of[scene] for scene in bank.scene_ids)
    sizes = sorted(per_sequence.values())
    return {
        "n_scenes": bank.n_scenes,
        "n_sequences": bank.n_sequences,
        "largest_sequence": sizes[-1] if sizes else 0,
        "median_sequence": sizes[len(sizes) // 2] if sizes else 0,
    }
