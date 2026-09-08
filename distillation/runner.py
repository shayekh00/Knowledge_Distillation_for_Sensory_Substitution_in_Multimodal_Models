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
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

import torch

from distillation.losses import (
    candidate_kd_loss,
    contrastive_loss,
    feature_transfer_loss,
    loca_kd_loss,
    masked_cross_entropy,
    shift_for_causal_lm,
    token_kd_loss,
    valid_answer_mask,
)
from distillation.xtoken import calibrate_topk_with_loca, projected_kl_loss

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

    def stage_one(self) -> "RecipeConfig":
        """The **F** half of a two-stage row: feature alignment, nothing else.

        §7.1 already describes D3–D9 as two-stage rows that "pay that cost
        roughly twice, since stage one is a separate training pass before S2
        begins," and §8.2 already defines S2 as vision-frozen. Those two
        statements together mean a row's feature objective belongs to its F
        stage, not to a single joint pass — which is also the author's own
        two-phase method (§13, 2026-09-07).

        Derived rather than declared as separate library rows on purpose. A row
        keeps one id and one declared feature objective, so `D8` stays "CE +
        X-Token + LoCa + cosine alignment" in the matrix, and the split into two
        passes is an implementation of that row rather than a renaming of it.
        This also avoids adding a field to this dataclass, which would change
        every KD row's run-id hash — `record_run.build_configuration` folds
        `resolved()` into the id, so a new field would silently orphan X2's
        completed run.
        """
        if self.feature_objective == "none":
            raise ValueError(
                f"recipe {self.recipe} declares no feature objective, so it has no "
                "stage one. Run it directly as a single-stage row.")
        return replace(
            self, stage="F", use_ce=False, kd_objective="none", use_loca=False,
            top_k=None, lambda_ce=0.0, lambda_kd=0.0,
            # Author decision, 2026-09-07: the merger is the model's own
            # vision->language projector, not part of the frozen pretrained
            # decoder — leaving it fixed caps how much the alignment loss can
            # actually reshape the vision->language interface, versus only
            # reshaping what happens *before* the merger. Verified the merger's
            # LoRA targets (`linear_fc1`/`linear_fc2`) are distinct from the 12
            # vision blocks' identically-named MLP layers before adding this —
            # a suffix-style match would have silently trained all 12 alongside
            # the merger asked for.
            trainable_modules=("vision_attention", "vision_merger"),
            parent_checkpoint=None,
            notes=f"stage one (F) of {self.recipe}: {self.feature_objective} alignment")

    def stage_one_p(self, p_lambda_kd: float = 0.1) -> "RecipeConfig":
        """The **P** half of a two-stage row (§8.2): feature alignment plus a
        small *raw* KD term, run on the same vision-only surface as F.

        This is D6's stage one — "as submitted": the rejected paper's method
        blended a small answer-distribution signal into the alignment pass
        itself, rather than keeping alignment and KD strictly separate the way
        D5's stage F does. Kept as its own method rather than a parameter on
        `stage_one()` because the two are answering different questions (D5:
        does clean alignment help; D6: does the original blended method), and
        collapsing them would make `stage_one()` silently change behaviour for
        every row that already uses it.

        Mechanically this needs no new plumbing: `compose_loss` already calls
        `adapter.student_logits` whenever `kd_objective` is set, regardless of
        `stage`, so leaving `kd_objective` in place (unlike `stage_one`, which
        clears it) is sufficient to make the KD term's gradient flow back
        through the frozen language decoder into the trainable vision surface
        — frozen means no optimizer step for those parameters, not no
        backward pass through them. `use_loca=False` is not a simplification;
        §8.2 defines P's own KD term as **raw** KD specifically ("Raw KD —
        teacher→student KL without gold-conditioned correction"), so LoCa is
        off here even on a row (D6) whose S2 stage uses it. `p_lambda_kd`
        defaults to 0.1 — an explicit, overridable weight for what §8.2 calls
        "a small raw KD term"; the protocol does not fix a number, so this is
        a documented author-facing choice, not a measured one.
        """
        if self.feature_objective == "none":
            raise ValueError(
                f"recipe {self.recipe} declares no feature objective, so it has no "
                "stage one (P or F).")
        if self.kd_objective == "none":
            raise ValueError(
                f"recipe {self.recipe} declares no kd_objective, so its stage one has "
                "nothing to make it P rather than F — use stage_one() instead.")
        return replace(
            self, stage="P", use_ce=False, use_loca=False,
            lambda_ce=0.0, lambda_kd=p_lambda_kd,
            trainable_modules=("vision_attention", "vision_merger"),
            parent_checkpoint=None,
            notes=f"stage one (P) of {self.recipe}: {self.feature_objective} alignment "
                  f"+ raw {self.kd_objective} KD (lambda_kd={p_lambda_kd})")

    def stage_two(self, parent_checkpoint: str) -> "RecipeConfig":
        """The **S2** half: the row's answer objectives, vision frozen.

        `feature_objective` is cleared because stage one already did the
        alignment and S2 freezes the vision encoder — leaving it set would
        recreate exactly the no-op `assert_trainable_surface_can_learn` refuses,
        a feature term whose gradient reaches nothing but the alignment head.
        The alignment is carried into this stage by the *weights*
        `parent_checkpoint` points at, not by re-running the loss.
        """
        if not parent_checkpoint:
            raise ValueError("stage two of a two-stage row needs its parent checkpoint")
        return replace(
            self, stage="S2", feature_objective="none", lambda_feature=0.0,
            parent_checkpoint=parent_checkpoint,
            notes=f"stage two (S2) of {self.recipe}, from {parent_checkpoint}")

    def is_two_stage(self) -> bool:
        """False for a `stage="joint"` row (D7, B3) even with a feature
        objective declared: joint means the feature term runs in the same
        single pass as everything else, on a surface that already includes
        `vision_attention` — the two-stage machinery (merge stage-F's adapter,
        freeze vision, run S2 from a parent checkpoint) does not apply and
        must not be forced onto it."""
        return self.stage != "joint" and self.feature_objective != "none"

    def assert_trainable_surface_can_learn(self) -> None:
        """Refuse a configuration whose objective cannot reach any parameter.

        Called at run launch rather than in `__post_init__` so the recipe library
        stays constructible and inspectable — but called before a single step, so
        a row that would train as if its distinguishing term were absent cannot
        quietly produce a results table.

        Measured, not theorised: the declared feature layer is a pooled *vision*
        representation and sits upstream of every language parameter, so with a
        language-only surface the feature loss reaches nothing but the alignment
        head — 0 of 48 LoRA tensors received gradient. That is the audit's B4
        failure mode, a term that appears active and is not.
        """
        if self.feature_objective != "none" and "vision_attention" not in self.trainable_modules:
            raise ValueError(
                f"recipe {self.recipe} declares feature_objective="
                f"{self.feature_objective!r} but its trainable surface is "
                f"{self.trainable_modules}, which contains no vision parameters. The "
                "feature loss would update only the alignment head and contribute "
                "nothing to the student, while still logging a falling loss. Either "
                "add 'vision_attention' to trainable_modules, or move the feature "
                "objective to this row's F stage and chain it into S2 via "
                "parent_checkpoint (experiment_protocol.md §13.1 point 5).")

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
                 xtoken_mapping=None, gold_teacher_lookup=None) -> tuple:
    """Assemble the total objective from the switches.

    Returns `(total, components)` where `components` is a dict of detached scalars
    for `training_metrics.csv`. Logging each component separately is what makes a
    later "the KD term was effectively off" diagnosis possible — the defect the
    audit found by reading, and which a component log would have surfaced.

    `gold_teacher_lookup` (`xtoken.build_student_to_teacher_lookup`'s output) is
    required when `config.use_loca` is set on an `xtoken` recipe (D2/D5/D8) —
    it is what let this branch ignore `use_loca` entirely before, since nothing
    forced a caller to supply gold's identity in teacher-vocabulary space.
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
            # Only the supervised answer positions are compared, and they are
            # selected here rather than downstream for two reasons. The teacher
            # cache stores exactly these positions (a dense cache over every
            # prompt and padding position would be ~200x larger for the same
            # signal), so `projected_kl_loss`'s row-count check fails against
            # anything else. And softmaxing all [B, L] positions over the full
            # vocabulary allocates a float32 copy of a tensor that is almost
            # entirely padding — the same shape of waste that OOM'd the cache
            # generator on a 24 GB card. Narrow first, then upcast.
            shifted_logits, shifted_labels = shift_for_causal_lm(student_logits, labels)
            answer_mask = valid_answer_mask(shifted_labels).reshape(-1)
            answer_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))[answer_mask]
            # Qwen3.5 pads its lm_head to a multiple of 256: 248,320 output
            # columns against a 248,077-token vocabulary. Those 243 extra rows
            # are never trained, so their logits are arbitrary rather than
            # small, and they are not tokens any tokenizer can produce. Slice
            # them off *before* the softmax so they are excluded from the
            # normaliser — softmaxing the padded width first would let
            # untrained columns take probability mass away from real tokens.
            # The teacher side needs no equivalent: its cached top-K ids were
            # measured to max out at 248,069, entirely inside the vocabulary,
            # with exactly zero mass on any padded id.
            vocabulary = xtoken_mapping.student_vocab_size
            if answer_logits.size(-1) < vocabulary:
                raise ValueError(
                    f"student emits {answer_logits.size(-1)} logits but the mapping "
                    f"expects at least {vocabulary}; the mapping belongs to a "
                    f"different tokenizer than this model")
            student_probs = torch.softmax(answer_logits[:, :vocabulary].float(), dim=-1)

            if config.use_loca:
                if gold_teacher_lookup is None:
                    raise ValueError(
                        "recipe declares use_loca on an xtoken objective (D2/D5/D8) "
                        "but no gold_teacher_lookup was supplied — this is exactly "
                        "the gap that let use_loca silently do nothing here before "
                        "this was wired up; build one with "
                        "xtoken.build_student_to_teacher_lookup(xtoken_mapping)")
                gold_student_ids = shifted_labels.reshape(-1)[answer_mask]
                gold_teacher_ids = gold_teacher_lookup[gold_student_ids]
                calibrated_probs, found_mask = calibrate_topk_with_loca(
                    topk_ids, topk_probs, gold_teacher_ids, alpha=config.loca_alpha)
                components["loca_gold_found_rate"] = float(found_mask.float().mean())
                if not found_mask.any():
                    raise ValueError(
                        f"LoCa found gold in the cached top-K for 0 of "
                        f"{found_mask.numel()} positions in this batch — either "
                        f"top_k={config.top_k} is too small for this recipe or "
                        f"something upstream (the mapping, the lookup, the cache) "
                        f"is misaligned. Refusing to train on an empty calibrated set.")
                kd = projected_kl_loss(topk_ids[found_mask], calibrated_probs,
                                       student_probs[found_mask], xtoken_mapping,
                                       temperature=config.kd_temperature)
            else:
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
            # The audit's B4 requirement: "log the number of distinct scenes per
            # candidate set". The historical contrastive term was identically
            # zero because its candidate set held one scene, and no logged
            # quantity would have revealed it.
            scenes = teacher.metadata.get("distinct_negative_scenes")
            if scenes is not None:
                components["negative_scenes"] = float(scenes)
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
        # trainable_modules is NOT the default here, and must not be: a pooled
        # *vision* feature loss has no gradient path to language attention, so
        # D0 with the default surface trains zero student parameters — measured
        # 0 of 48 LoRA tensors receiving gradient, with only the alignment head
        # updating. It would still log a falling feature loss and save a
        # checkpoint byte-identical in behaviour to the base model. §8.2's
        # stage-F definition ("selected vision parameters update; projector and
        # language parameters frozen") is what this now declares.
        "D0": RecipeConfig(recipe="D0", stage="F", use_ce=False,
                           feature_objective="contrastive",
                           trainable_modules=("vision_attention", "vision_merger")),
        "D1": RecipeConfig(recipe="D1", stage="S2", use_ce=True, kd_objective="xtoken",
                           top_k=top_k),
        "D2": RecipeConfig(recipe="D2", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, top_k=top_k),
        "D3": RecipeConfig(recipe="D3", stage="S2", use_ce=True,
                           feature_objective="contrastive"),
        # Not run — author decision, experiment_protocol.md §13 2026-09-07: the
        # manuscript presents the F->S2 pipeline as one distillation method and
        # does not attribute D5's gain between stage F and stage S2, so the row
        # that would answer that (CE-only on aligned vision) is left declared
        # but unexecuted.
        "D4": RecipeConfig(recipe="D4", stage="S2", use_ce=True, kd_objective="xtoken",
                           feature_objective="contrastive", top_k=top_k),
        "D5": RecipeConfig(recipe="D5", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, feature_objective="contrastive", top_k=top_k),
        # Same declared S2 as D5 — D5 and D6 are distinguished operationally, by
        # which stage-one derivation is run (`stage_one()` for D5, `stage_one_p()`
        # for D6), not by any field here. See RecipeConfig.stage_one_p().
        "D6": RecipeConfig(recipe="D6", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, feature_objective="contrastive", top_k=top_k),
        # Joint single pass (§13.1 point 5's rejected option (b), used deliberately
        # here rather than for D3/D5/D8): both surfaces trainable at once, feature
        # term never cleared. `is_two_stage()` returns False because stage="joint".
        "D7": RecipeConfig(recipe="D7", stage="joint", use_ce=True, kd_objective="xtoken",
                           use_loca=True, feature_objective="contrastive", top_k=top_k,
                           trainable_modules=("language_attention", "vision_attention")),
        "D8": RecipeConfig(recipe="D8", stage="S2", use_ce=True, kd_objective="xtoken",
                           use_loca=True, feature_objective="cosine", top_k=top_k),
        # §8.1's strict label-access rule (no CE, no LoCa, no gold prefixes) is
        # a *data-path* requirement this config cannot enforce by itself: the
        # gold answer column must be removed before the training/cache
        # interface and replaced by a teacher-generated prefix, which needs a
        # teacher-completion cache that does not exist yet (a new artifact,
        # distinct from build_teacher_cache.py's top-K logit cache). This entry
        # is the row's declared shape, not a run-ready recipe — do not launch
        # it until that cache and its batch-builder path are built.
        "D9": RecipeConfig(recipe="D9", stage="S2", use_ce=False, kd_objective="xtoken",
                           feature_objective="contrastive", top_k=top_k,
                           notes="BLOCKED: needs a teacher-generated-prefix cache; "
                                 "see experiment_protocol.md §8.1"),
        # Distillation-mode ladder
        "X0": RecipeConfig(recipe="X0", stage="S2", use_ce=True, kd_objective="sequence"),
        "X1": RecipeConfig(recipe="X1", stage="S2", use_ce=True, kd_objective="candidate"),
        "X2": RecipeConfig(recipe="X2", stage="S2", use_ce=True, kd_objective="xtoken",
                           top_k=top_k),
        "X3": RecipeConfig(recipe="X3", stage="S2", use_ce=False, kd_objective="xtoken",
                           top_k=top_k, lambda_ce=0.0),
        "X5": RecipeConfig(recipe="X5", stage="S2", use_ce=True, kd_objective="token"),
    }
