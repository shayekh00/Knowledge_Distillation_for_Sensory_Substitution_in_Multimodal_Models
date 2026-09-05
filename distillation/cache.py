"""Teacher-signal cache keys and the per-run manifest (plan §8.2, §15).

Two jobs, both about making a result traceable to the exact thing that produced
it.

**Cache keys.** A teacher cache is only reusable when everything that determined
its contents is unchanged. The plan enumerates those fields, and the protocol adds
top-K and both tokenizer revisions for X-Token. Getting this wrong is silent: a
cache reused across a precision or K change yields plausible numbers computed
against the wrong targets, and nothing crashes. So the key is computed from a
declared field set, stored beside the data, and **verified on read**.

**Run manifests.** Every result, checkpoint, and prediction file carries a run id,
and the manifest records what the run actually was. R1 could not reconcile the
previous submission's headline with its ablations; the manifest is what makes that
class of question answerable next time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field

RUN_ID_PATTERN = re.compile(r"^\d{8}-[a-z0-9]+-[A-Za-z0-9]+-s\d+-[0-9a-f]{8}$")

DISTILLATION_MODES = ("none", "sequence", "candidate", "xtoken", "hybrid")

# Everything that determines a cached teacher signal. Adding a field here is a
# deliberate act: it invalidates every existing cache, which is the point.
CACHE_KEY_FIELDS = (
    "dataset_version",
    "split",
    "teacher_model",
    "teacher_revision",
    "processor_revision",
    "precision",
    "prompt_hash",
    "prefix_source",
    "depth_transform",
    "rgb_transform",
    "signal_kind",
    "top_k",
    "temperature",
    "teacher_tokenizer_revision",
    "student_tokenizer_revision",
    "xtoken_mapping_hash",
    "feature_layer",
    "crop_aggregation",
)

SIGNAL_KINDS = ("topk_logits", "candidate_scores", "generated_text", "pooled_features")


@dataclass(frozen=True)
class CacheKey:
    """The identity of a teacher cache. Two caches with equal keys are
    interchangeable; two with different keys are different experiments."""
    fields: dict

    def __post_init__(self):
        unknown = set(self.fields) - set(CACHE_KEY_FIELDS)
        if unknown:
            raise ValueError(f"unknown cache key field(s): {sorted(unknown)}")
        missing = {"dataset_version", "split", "teacher_model", "precision", "signal_kind"} \
            - set(self.fields)
        if missing:
            raise ValueError(f"cache key is missing required field(s): {sorted(missing)}")
        kind = self.fields["signal_kind"]
        if kind not in SIGNAL_KINDS:
            raise ValueError(f"unknown signal_kind {kind!r}; expected one of {SIGNAL_KINDS}")
        if kind == "topk_logits" and not self.fields.get("top_k"):
            raise ValueError(
                "top_k must be set for a topk_logits cache — it determines the "
                "contents, so omitting it would let two different caches share a key")

    def digest(self) -> str:
        canonical = json.dumps(
            {name: self.fields.get(name) for name in CACHE_KEY_FIELDS},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def describe(self) -> dict:
        return {name: self.fields.get(name) for name in CACHE_KEY_FIELDS}


def cache_directory(root: str, key: CacheKey) -> str:
    return os.path.join(root, f"{key.fields['signal_kind']}_{key.digest()}")


def write_cache_key(directory: str, key: CacheKey) -> str:
    """Persist the key beside its data so a later read can verify it."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "cache_key.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"digest": key.digest(), "fields": key.describe()}, handle, indent=2)
    return path


def verify_cache_key(directory: str, expected: CacheKey) -> None:
    """Refuse a cache that was not built for this configuration.

    Raises with the differing fields named, because "cache mismatch" alone sends
    the reader hunting through two configs by hand.
    """
    path = os.path.join(directory, "cache_key.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{directory} has no cache_key.json; refusing to use an unidentified cache")
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored.get("digest") == expected.digest():
        return
    differing = [name for name in CACHE_KEY_FIELDS
                 if stored.get("fields", {}).get(name) != expected.describe().get(name)]
    raise ValueError(
        f"teacher cache at {directory} does not match this configuration. "
        f"Differing field(s): {differing}. Regenerate the cache rather than reusing it — "
        "a mismatched cache produces plausible numbers against the wrong targets.")


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """The per-run record required by §15's artifact contract."""
    run_id: str
    recipe: str
    seed: int
    dataset_version: str
    student_model: str
    student_revision: str
    trainable_modules: list
    distillation_mode: str = "none"
    teacher_model: str | None = None
    teacher_revision: str | None = None
    precision: str = "bf16"
    pilot: bool = True
    teacher_tokenizer_revision: str | None = None
    student_tokenizer_revision: str | None = None
    xtoken_mapping_hash: str | None = None
    top_k: int | None = None
    mean_omitted_teacher_mass: float | None = None
    loss_weights: dict = field(default_factory=dict)
    cache_digests: dict = field(default_factory=dict)
    parent_checkpoint: str | None = None
    evaluator_hash: str | None = None
    prompt_hash: str | None = None
    notes: str = ""

    def validate(self) -> None:
        if not RUN_ID_PATTERN.match(self.run_id):
            raise ValueError(
                f"run_id {self.run_id!r} does not match "
                "{YYYYMMDD}-{pair}-{recipe}-s{seed}-{cfg8}")
        if self.distillation_mode not in DISTILLATION_MODES:
            raise ValueError(
                f"unknown distillation_mode {self.distillation_mode!r}; "
                f"expected one of {DISTILLATION_MODES}")
        if self.distillation_mode != "none" and not self.teacher_model:
            raise ValueError("a distillation run must name its teacher_model")
        if self.distillation_mode in ("xtoken", "hybrid"):
            for name in ("teacher_tokenizer_revision", "student_tokenizer_revision",
                         "xtoken_mapping_hash", "top_k"):
                if getattr(self, name) in (None, ""):
                    raise ValueError(
                        f"X-Token run must record {name}: it is part of the frozen "
                        "target source, and two runs differing in it are not comparable")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, run_directory: str) -> str:
        os.makedirs(run_directory, exist_ok=True)
        path = os.path.join(run_directory, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        return path


def make_run_id(date: str, pair: str, recipe: str, seed: int,
                resolved_configuration: dict) -> str:
    """Build the protocol's run id.

    The trailing hash is taken over the fully resolved configuration, so two runs
    claiming the same recipe with different settings cannot collide — silent
    setting drift becomes a visibly different id.
    """
    canonical = json.dumps(resolved_configuration, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:8]
    run_id = f"{date}-{pair}-{recipe}-s{seed}-{digest}"
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"generated run_id {run_id!r} is malformed; check pair/recipe slugs")
    return run_id
