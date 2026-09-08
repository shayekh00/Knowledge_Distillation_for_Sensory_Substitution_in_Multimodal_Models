"""Read a cached teacher signal back as `TeacherSignals` (plan §8.2).

The counterpart to `build_teacher_cache.py`. That script runs the teacher alone
and unloads it; this one turns what it wrote into the tensors `compose_loss`
consumes, so the student never shares the GPU with a teacher.

Two invariants matter here and both are enforced rather than assumed:

* **The cache key is verified on read.** `cache.verify_cache_key` refuses a cache
  built for a different precision, K, prompt, or tokenizer revision. A silently
  reused cache produces plausible numbers computed against the wrong targets and
  nothing crashes, which is the failure mode the key exists to prevent.
* **Row order matches the loss.** `projected_kl_loss` compares teacher row *i*
  against student row *i* and only checks that the counts agree, so an ordering
  mistake here would train against mismatched targets without ever raising.
  Rows are therefore concatenated in batch order, and each example's own row
  count is checked against the student's supervised-position count for that
  example by `assert_rows_align`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from distillation.cache import CacheKey, verify_cache_key
from distillation.features import pool_per_image, sample_negative_bank
from distillation.losses import shift_for_causal_lm, valid_answer_mask
from distillation.runner import TeacherSignals


class TeacherCache:
    """Random access to one cache directory, verified against its key."""

    def __init__(self, directory: str, expected_key: CacheKey):
        verify_cache_key(directory, expected_key)
        self.directory = directory
        self.key = expected_key

    def has(self, question_id: str) -> bool:
        return os.path.isfile(os.path.join(self.directory, f"{question_id}.npz"))

    def load_rows(self, question_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """One example's ``([n, K] ids, [n, K] probs)``, n = its answer positions."""
        path = os.path.join(self.directory, f"{question_id}.npz")
        if not os.path.isfile(path):
            raise KeyError(
                f"{question_id} is absent from {self.directory}. The cache was built "
                f"over a different row set — regenerate it for this split rather than "
                f"training on a silently smaller one.")
        with np.load(path) as payload:
            ids = torch.from_numpy(payload["topk_ids"].astype(np.int64))
            probs = torch.from_numpy(payload["topk_probs"].astype(np.float32))
        return ids, probs

    def signals_for(self, question_ids, device=None) -> TeacherSignals:
        """The batch's teacher rows, concatenated in the order given.

        That order must be the order the student's supervised positions appear
        in after `shift_for_causal_lm` + `valid_answer_mask`, which is row-major
        over the batch — the same order `build_teacher_cache.py` wrote them.
        """
        pairs = [self.load_rows(question_id) for question_id in question_ids]
        ids = torch.cat([pair[0] for pair in pairs])
        probs = torch.cat([pair[1] for pair in pairs])
        if device is not None:
            ids, probs = ids.to(device), probs.to(device)
        return TeacherSignals(
            topk_ids=ids, topk_probs=probs,
            metadata={"rows_per_example": [pair[0].size(0) for pair in pairs],
                      "cache_digest": self.key.digest()})


class GeneratedTextCache:
    """Random access to a `generated_text` cache — the teacher's own
    free-generated completions, from `build_teacher_generation_cache.py`.

    D9's substitute for `row["answer"]` (§8.1): every other recipe's KD term
    forces the gold answer as a teaching-forcing prefix even with CE off, which
    leaks the label. D9 forces this cache's text instead, so `answers[qid]`
    below is the only thing that ever stands in for `row["answer"]` on this
    row's training/cache-building path.

    One consolidated JSON table, like `FeatureCache`'s one consolidated table
    and for the same reason: the payload is small (a few words per row, a few
    thousand rows) and needs random access by question id, not a per-batch
    stream.
    """

    def __init__(self, directory: str, expected_key: CacheKey):
        verify_cache_key(directory, expected_key)
        self.directory = directory
        self.key = expected_key
        path = os.path.join(directory, "completions.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} is missing. A generated_text cache is one consolidated "
                f"JSON table, not per-example files — run "
                f"build_teacher_generation_cache.py for this key.")
        with open(path, encoding="utf-8") as handle:
            self.completions: dict = json.load(handle)

    def has(self, question_id: str) -> bool:
        return question_id in self.completions

    def answers_for(self, question_ids) -> dict:
        """`{question_id: generated text}` for exactly the ids given — the
        shape `build_batch_with_answers` needs. Raises rather than silently
        substituting anything (gold included) for a row this cache never
        generated a completion for."""
        missing = [qid for qid in question_ids if qid not in self.completions]
        if missing:
            raise KeyError(
                f"{len(missing)} question id(s) absent from {self.directory}, e.g. "
                f"{missing[0]!r}. The generation cache was built over a different "
                f"row set — regenerate it for this split rather than training "
                f"with a gold fallback, which would defeat §8.1's label-access rule.")
        return {qid: self.completions[qid] for qid in question_ids}


def assert_rows_align(teacher: TeacherSignals, labels: torch.Tensor) -> None:
    """Fail loudly when the cache and the batch disagree about answer positions.

    `projected_kl_loss` only compares row *counts*; if the cache was built from a
    different prompt, tokenizer, or answer field, the counts can still match by
    coincidence per batch while every row is the wrong target. Comparing
    per-example counts makes that far harder to miss, and costs nothing.
    """
    expected = valid_answer_mask(labels[:, 1:]).sum(dim=1).tolist()
    actual = teacher.metadata.get("rows_per_example")
    if actual is None:
        raise ValueError("teacher signals carry no per-example row counts to check")
    if list(actual) != expected:
        raise ValueError(
            f"teacher cache rows per example {list(actual)} do not match the batch's "
            f"supervised positions {expected}. The cache was built from a different "
            f"prompt, tokenizer, or answer column — regenerate it rather than "
            f"training against misaligned targets.")


def resolve_visual_tower(model):
    """Find the vision tower through however many wrappers are in the way.

    `model.model.visual` is right for a bare `Qwen3_5ForConditionalGeneration`
    and wrong once PEFT wraps it: `PeftModel.model` is the LoRA wrapper, so the
    same expression lands one level short and raises `AttributeError` on
    `.visual`. Training always wraps, so a hardcoded path works in a test and
    fails on the first real step — which is exactly what it did.

    Walking is done once, in the adapter's constructor, so a model this cannot
    handle fails at construction with a readable message instead of mid-step.
    """
    current = model
    for _ in range(6):
        if hasattr(current, "visual"):
            return current.visual
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        if hasattr(current, "model"):
            current = current.model
            continue
        break
    raise ValueError(
        f"no `visual` module found under {type(model).__name__}. A feature objective "
        "needs the vision tower; guessing an attribute path here is how the pooled "
        "feature silently becomes something else.")


class QwenAdapter:
    """`ModelAdapter` for the Qwen3.5 pair (plan §9.3's primary pair).

    Deliberately thin: `compose_loss` is a pure function of tensors, so the only
    model-family-specific knowledge is how to get logits out and which parameters
    a stage trains. A second family is added by writing another of these, not by
    touching any loss.
    """

    def __init__(self, model, alignment_head=None, spatial_merge_size: int | None = None):
        self.model = model
        self.alignment_head = alignment_head
        if spatial_merge_size is None:
            spatial_merge_size = getattr(
                getattr(getattr(model, "config", None), "vision_config", None),
                "spatial_merge_size", None)
            if spatial_merge_size is None:
                raise ValueError(
                    "spatial_merge_size could not be read from the model config and was "
                    "not supplied. Guessing it would mis-split the packed vision "
                    "sequence, which changes the pooled features without any error.")
        self.spatial_merge_size = spatial_merge_size
        self._visual = None   # resolved lazily: a logits-only row never needs it

    def student_logits(self, batch):
        inputs = {key: value for key, value in batch.items()
                  if key not in ("labels", "question_ids")}
        return self.model(**inputs).logits

    def student_features(self, batch):
        """Pooled, projected vision feature per example — `[B, teacher_dim]`.

        Runs the vision tower directly rather than reusing the one already
        computed inside `student_logits`. That is a deliberate cost: for a
        stage-F row (D0) `compose_loss` never calls `student_logits` at all, so
        a reuse-only design would have nothing to read; and for CE+feature rows
        the obvious optimisation — cache the merger output from the last forward
        and reuse it — can silently serve *stale* features if a freed
        `pixel_values` and a new one land on the same address with the same
        shape. Wrong-but-plausible features produce a wrong loss and no error,
        which is the failure class this module exists to refuse. The duplicated
        work is the 12-block student vision tower (measured 1.85 GB peak
        standalone), not the language model.

        The alignment head is required: this pair's widths differ (student 1024,
        teacher 4096) and both feature losses demand equal widths, so without it
        the failure would surface as a shape error deep inside the loss.
        """
        if self.alignment_head is None:
            raise ValueError(
                "a feature objective needs an AlignmentHead: the student's pooled "
                "feature is 1024-wide and the cached teacher's is 4096-wide, and both "
                "feature losses require equal widths. Construct QwenAdapter with "
                "alignment_head=AlignmentHead(student_dim, teacher_dim).")
        if "pixel_values" not in batch or "image_grid_thw" not in batch:
            raise ValueError(
                "batch has no pixel_values/image_grid_thw, so no vision feature exists "
                "to align. A text-only batch cannot carry a feature objective.")
        if self._visual is None:
            self._visual = resolve_visual_tower(self.model)
        merged = self._visual(batch["pixel_values"],
                              grid_thw=batch["image_grid_thw"]).pooler_output
        pooled = pool_per_image(merged, batch["image_grid_thw"],
                                self.spatial_merge_size, normalize=True)
        return self.alignment_head(pooled)

    def labels(self, batch):
        return batch["labels"]

    def trainable_parameters(self, config):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]


class FeatureCache:
    """Random access to a `pooled_features` cache, verified against its key.

    The counterpart to `build_feature_cache.py`, and the reason D0/D3/D5/D8 can
    run at all: `compose_loss` reads `teacher.features` from cached signals, and
    nothing populated that field before.

    Unlike `TeacherCache` this holds the whole table in memory. It is small — one
    vector per image, 4,187 images x 4096 float16 for the train split — and the
    contrastive objective needs random access across *all* scenes on every step
    to draw negatives, so streaming it from disk per batch would be the wrong
    trade entirely.
    """

    def __init__(self, directory: str, expected_key: CacheKey):
        verify_cache_key(directory, expected_key)
        self.directory = directory
        self.key = expected_key
        path = os.path.join(directory, "feature_table.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} is missing. A pooled_features cache is one consolidated "
                f"table, not per-example files — run build_feature_cache.py for this key.")
        with np.load(path, allow_pickle=False) as payload:
            # float16 on disk, float32 in memory: the losses normalise and take
            # inner products, and float16 accumulation over a 4096-wide dot
            # product loses precision that costs nothing to avoid here.
            self.features = torch.from_numpy(payload["features"].astype(np.float32))
            self.image_ids = [str(value) for value in payload["image_ids"]]
            self.sequence_ids = [str(value) for value in payload["sequence_ids"]]
        if not (len(self.image_ids) == len(self.sequence_ids) == self.features.size(0)):
            raise ValueError(
                f"feature table is inconsistent: {self.features.size(0)} feature rows, "
                f"{len(self.image_ids)} image ids, {len(self.sequence_ids)} sequence ids")
        self.row_of_image = {image_id: row for row, image_id in enumerate(self.image_ids)}
        if len(self.row_of_image) != len(self.image_ids):
            raise ValueError(
                "the feature table contains duplicate image_ids, so a positive lookup "
                "would be ambiguous. Regenerate it — build_feature_cache.py dedupes "
                "by first appearance.")

    @property
    def feature_dim(self) -> int:
        return self.features.size(-1)

    def signals_for(self, rows, bank_size: int = 255, generator=None,
                    device=None) -> TeacherSignals:
        """Positives for this batch plus a shared negative bank.

        `rows` are the batch's dataset rows, in the batch's own order, because
        `contrastive_loss` and `feature_transfer_loss` both compare student row
        *i* against teacher row *i*.

        The bank excludes **every sequence present in the batch**, not just each
        example's own. The bank is shared across the batch, so a scene that is a
        room neighbour of example *j* would otherwise appear among example *i*'s
        negatives while being a near-duplicate of *j*'s positive — a false
        negative that the audit's exclusion rule exists to prevent.
        """
        missing = [row["image_id"] for row in rows if row["image_id"] not in self.row_of_image]
        if missing:
            raise KeyError(
                f"{len(missing)} image(s) absent from the feature cache, e.g. "
                f"{missing[0]!r}. The cache was built over a different split or row "
                f"set — regenerate it rather than training on a silently smaller one.")
        positive_rows = torch.tensor([self.row_of_image[row["image_id"]] for row in rows],
                                     dtype=torch.long)
        positives = self.features[positive_rows]
        bank, distinct_scenes = sample_negative_bank(
            self.features, self.sequence_ids,
            exclude_sequences={row["sequence_id"] for row in rows},
            size=bank_size, generator=generator)
        if device is not None:
            positives, bank = positives.to(device), bank.to(device)
        return TeacherSignals(
            features=positives, negative_bank=bank,
            metadata={"cache_digest": self.key.digest(),
                      "distinct_negative_scenes": distinct_scenes,
                      "feature_layer": self.key.fields.get("feature_layer"),
                      "excluded_sequences": len({row["sequence_id"] for row in rows})})
