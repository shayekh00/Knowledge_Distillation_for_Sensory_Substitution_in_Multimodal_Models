"""One shared experiment runner with explicit switches (plan §7.2).

The legacy tree kept a separate copied training program per ablation
(`knowledge_distillation7b_logit_based/`, `_feature_based/`, `_double_trouble/`
phases 1-3, …). That is how D1 and D4 come to differ in ways nobody intended, and
it is why the historical rows cannot be reconciled. There is one runner here, and
an ablation is a **configuration**, not a fork.

Three pieces:

* :class:`RecipeConfig` — every switch, resolvable to a dict that hashes into the
  run id, so silent setting drift is visible.
* :class:`ModelAdapter` — the only place model-family details live. Losses stay
  pure tensor functions; the adapter supplies tensors.
* :func:`compose_loss` — assembles the objective from the switches, and refuses
  configurations that cannot mean what they say.

Deliberately not a plugin system and not an inheritance hierarchy. The plan asks
for an Adapter to separate model families and a small Pipeline for
data -> teacher cache -> training -> predictions -> scoring; anything more is
maintenance without benefit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol

import torch

from distillation.losses import (
    candidate_kd_loss,
    contrastive_loss,
    feature_transfer_loss,
    loca_kd_loss,
    masked_cross_entropy,
    token_kd_loss,
)
from distillation.xtoken import projected_kl_loss

STAGES = ("F", "P", "S2", "joint")
KD_OBJECTIVES = ("none", "sequence", "candidate", "token", "xtoken")
FEATURE_OBJECTIVES = ("none", "cosine", "mse", "contrastive")


@dataclass
class RecipeConfig:
    """Every switch that defines a training row.

    Field names are deliberately close to the §9.2 matrix vocabulary so a row id
    (`D5`, `X2`) maps onto a config without translation.
    """
    recipe: str
    stage: str = "S2"
    seed: int = 17

    # Objective switches
    use_ce: bool = True
    kd_objective: str = "none"
    feature_objective: str = "none"
    use_loca: bool = False

    # Weights
    lambda_ce: float = 1.0
    lambda_kd: float = 1.0
    lambda_feature: float = 1.0

    # Temperatures, kept separate on purpose: a value below 1 sharpens, and the
    # contrastive temperature is a different quantity from the KD temperature.
    kd_temperature: float = 1.0
    contrastive_temperature: float = 0.07
    loca_alpha: float = 0.8

    # X-Token
    top_k: int | None = None
    xtoken_mapping_hash: str | None = None

    # Trainable surface
    trainable_modules: tuple = ("language_attention",)
    lora_rank: int = 16

    # Provenance
    pilot: bool = True
    parent_checkpoint: str | None = None
    notes: str = ""

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {STAGES}")
        if self.kd_objective not in KD_OBJECTIVES:
            raise ValueError(
                f"unknown kd_objective {self.kd_objective!r}; expected {KD_OBJECTIVES}")
        if self.feature_objective not in FEATURE_OBJECTIVES:
            raise ValueError(
                f"unknown feature_objective {self.feature_objective!r}; "
                f"expected {FEATURE_OBJECTIVES}")
        if not self.use_ce and self.kd_objective == "none" and self.feature_objective == "none":
            raise ValueError(
                f"recipe {self.recipe!r} has no objective at all: CE off, no KD, no "
                "feature loss. This is a configuration error, not a control.")
        if self.use_loca and self.kd_objective not in ("token", "xtoken"):
            raise ValueError(
                "LoCa is defined over a single categorical distribution with a gold "
                "class. It cannot be composed with candidate or sequence KD without a "
                "derivation that does not exist (NEW_SUBMISSION.md §7.3).")
        if self.kd_objective == "xtoken" and self.top_k is None:
            raise ValueError("an X-Token recipe must declare top_k; it determines the targets")
        if self.stage == "F" and self.use_ce:
            raise ValueError(
                "stage F is feature alignment only — projector and language parameters "
                "are frozen, so a CE term would have nothing to train")

    def label_exposed(self) -> bool:
        """Whether this row sees ground-truth answers in any form.

        Removing CE does not by itself make a run label-free: LoCa consumes gold,
        and so do gold answer prefixes (§9.4). Used to keep the §8 label-access
        inventory honest rather than trusting a row's name.
        """
        return self.use_ce or self.use_loca

    def resolved(self) -> dict:
        """Fully resolved settings — the dict that hashes into the run id."""
        return asdict(self)

    def to_yaml_like(self) -> str:
        return json.dumps(self.resolved(), indent=2, sort_keys=True)


class ModelAdapter(Protocol):
    """The only model-family-specific surface.

    Keeping this small is what lets a second family be added without touching the
    losses: an adapter supplies tensors, and every objective below is a pure
    function of tensors.
    """

    def student_logits(self, batch) -> torch.Tensor: ...
    def student_features(self, batch) -> torch.Tensor: ...
    def labels(self, batch) -> torch.Tensor: ...
    def trainable_parameters(self, config: RecipeConfig) -> list: ...


@dataclass
class TeacherSignals:
    """Cached teacher outputs for one batch. Never a live teacher model.

    §8.2 runs the teacher alone, caches, and unloads it, so the student never
    shares the GPU with it. Which fields are populated depends on the objective.
    """
    logits: torch.Tensor | None = None
    candidate_scores: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    topk_probs: torch.Tensor | None = None
    features: torch.Tensor | None = None
    negative_bank: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)


def _require(value, name: str, objective: str):
    if value is None:
        raise ValueError(
            f"{objective} KD requires teacher {name}, which the cache did not supply. "
            "Regenerate the cache for this objective rather than falling back silently.")
    return value


def compose_loss(config: RecipeConfig, adapter: ModelAdapter, batch,
                 teacher: TeacherSignals, student_candidate_scores=None,
                 xtoken_mapping=None) -> tuple:
    """Assemble the total objective from the switches.

    Returns `(total, components)` where `components` is a dict of detached scalars
    for `training_metrics.csv`. Logging each component separately is what makes a
    later "the KD term was effectively off" diagnosis possible — the defect the
    audit found by reading, and which a component log would have surfaced.
    """
    components: dict[str, float] = {}
    total = torch.zeros((), dtype=torch.float32)

    labels = adapter.labels(batch)
    student_logits = None

    if config.use_ce:
        student_logits = adapter.student_logits(batch)
        ce = masked_cross_entropy(student_logits, labels)
        total = total + config.lambda_ce * ce
        components["ce"] = float(ce.detach())

    if config.kd_objective != "none" and config.kd_objective != "sequence":
        if student_logits is None:
            student_logits = adapter.student_logits(batch)

        if config.kd_objective == "token":
            teacher_logits = _require(teacher.logits, "logits", "token")
            kd = (loca_kd_loss(teacher_logits, student_logits, labels,
                               temperature=config.kd_temperature, alpha=config.loca_alpha)
                  if config.use_loca else
                  token_kd_loss(teacher_logits, student_logits, labels,
                                temperature=config.kd_temperature))
        elif config.kd_objective == "xtoken":
            if xtoken_mapping is None:
                raise ValueError("an X-Token recipe needs its vocabulary mapping")
            topk_ids = _require(teacher.topk_ids, "topk_ids", "X-Token")
            topk_probs = _require(teacher.topk_probs, "topk_probs", "X-Token")
            student_probs = torch.softmax(
                student_logits.reshape(-1, student_logits.size(-1)).float(), dim=-1)
            kd = projected_kl_loss(topk_ids, topk_probs, student_probs, xtoken_mapping,
                                   temperature=config.kd_temperature)
        else:                                            # candidate
            teacher_scores = _require(teacher.candidate_scores, "candidate_scores", "candidate")
            if student_candidate_scores is None:
                raise ValueError("candidate KD requires student candidate scores")
            kd = candidate_kd_loss(teacher_scores, student_candidate_scores,
                                   temperature=config.kd_temperature)
        total = total + config.lambda_kd * kd
        components["kd"] = float(kd.detach())

    if config.kd_objective == "sequence":
        # Sequence-level distillation is CE against the teacher's generated text;
        # the adapter supplies those as the batch labels, so there is no separate
        # term. Recorded explicitly so the mode is visible in the metrics.
        components["kd_mode"] = 0.0

    if config.feature_objective != "none":
        student_features = adapter.student_features(batch)
        teacher_features = _require(teacher.features, "features", config.feature_objective)
        if config.feature_objective == "contrastive":
            bank = _require(teacher.negative_bank, "negative_bank", "contrastive")
            feature = contrastive_loss(student_features, teacher_features, bank,
                                       temperature=config.contrastive_temperature)
        else:
            feature = feature_transfer_loss(student_features, teacher_features,
                                            kind=config.feature_objective)
        total = total + config.lambda_feature * feature
        components["feature"] = float(feature.detach())

    components["total"] = float(total.detach())
    return total, components


# ---------------------------------------------------------------------------
# The §9.2 matrix as configurations
# ---------------------------------------------------------------------------

def recipe_library(top_k: int = 4096) -> dict:
    """The experiment matrix as configs, so a row is selected rather than coded.

    Every row here maps to an ID in `NEW_SUBMISSION.md` §9.2. Adding a row to the
    paper means adding it here, which keeps the matrix and the code from drifting.
    """
    return {
        "B3": RecipeConfig(recipe="B3", stage="joint", use_ce=True),
        "B4": RecipeConfig(recipe="B4", stage="S2", use_ce=True),
        "D0": RecipeConfig(recipe="D0", stage="F", use_ce=False,
                           feature_objective="contrastive"),
        "D1": RecipeConfig(recipe="D1", stage="S2", use_ce=True, kd_objective="xtoken",
                           top_k=top_k),
        "D2": RecipeConfig(recipe="D2", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, top_k=top_k),
        "D3": RecipeConfig(recipe="D3", stage="S2", use_ce=True,
                           feature_objective="contrastive"),
        "D5": RecipeConfig(recipe="D5", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, feature_objective="contrastive", top_k=top_k),
        "D8": RecipeConfig(recipe="D8", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, feature_objective="cosine", top_k=top_k),
        # Distillation-mode ladder
        "X0": RecipeConfig(recipe="X0", stage="S2", use_ce=True, kd_objective="sequence"),
        "X1": RecipeConfig(recipe="X1", stage="S2", use_ce=True, kd_objective="candidate"),
        "X2": RecipeConfig(recipe="X2", stage="S2", use_ce=True, kd_objective="xtoken",
                           top_k=top_k),
        "X3": RecipeConfig(recipe="X3", stage="S2", use_ce=False, kd_objective="xtoken",
                           top_k=top_k, lambda_ce=0.0),
        "X5": RecipeConfig(recipe="X5", stage="S2", use_ce=True, kd_objective="token"),
    }
