# Implementation audit — data path and losses

**Status:** Phase 4 findings of `docs/NEW_SUBMISSION.md` §19 · feeds Gate **G3**
**Created:** 2026-09-05
**Companion to:** [`experiment_protocol.md`](experiment_protocol.md)

Every defect below was **reproduced**, not inferred from reading. Each entry gives
the mechanism, the evidence, and the requirement it places on the replacement
implementation.

> **Scope.** The audited code is the legacy OneVision path. `NEW_SUBMISSION.md`
> §8.1 replaces that portfolio, so these are **requirements for the new shared
> runner** (§7.2), not a patch list for code that will be retired. They are
> recorded because the same mistakes are easy to reproduce in a rewrite, and
> because several bear directly on how the historical numbers should be read.
>
> **Causal claims are withheld.** Per §7.1, a defect explains a score only when
> the corresponding historical configuration is re-executed. Nothing below is
> asserted as the cause of any published number.

---

## A. Data path

### A1 — The augmentation flag has no effect, and *no* transform reaches the model

`dataset/dataloader/OneVision/CustomSUNRGBDDatasetOneVision.py:199-212`

```python
if self.augmentation:
    rgb_image = self.rgb_augmentations(rgb_image)   # both branches
else:
    rgb_image = self.rgb_augmentations(rgb_image)   # are identical
depth_image = self.depth_preprocessing(depth_image)
...
return question, answer, rgb_image_np, depth_image_array, idx
```

Two separate faults:

1. **Both branches call the same pipeline**, so `augmentation=True/False` is inert.
2. **The transformed tensors are discarded.** `rgb_image` and `depth_image` are
   computed and then never returned — the function returns `rgb_image_np` and
   `depth_image_array`, the raw arrays captured *before* any transform (lines
   193, 196). The augmentation *and* the `Normalize(...)` are thrown away; the
   model receives raw uint8 arrays.

**Consequence.** No augmentation was applied in either arm, so an
augmentation-on/off ablation over this loader compares two identical conditions.
This is the mechanism behind R1's "augmentation configuration contradicts the
ablation".

**Requirement.** The tensor actually consumed by the processor must be the one the
test asserts on. Assert on the collated batch, never on an intermediate.

### A2 — Depth decoding does not match the generator, and diverges on `xtion`

`CustomSUNRGBDDatasetOneVision.py:86-94` opens the depth PNG and min-max
normalizes the stored integers per image. The v2 pipeline that produced the gold
answers instead applies the official SUN RGB-D 16-bit rotation
(`build_index.py:114-118`, `depth_utils.py:18-22`):

```
rotated = (raw >> 3) | (raw << 13)      # low 3 bits move to the TOP
metres  = clip(rotated / 1000, 0, 8.0)
```

Measured on the released test split:

| Sensor | Frames sampled | Rotation ≠ divide-by-8 | Ordering agreement |
|---|---:|---:|---:|
| kv1 | 12 | 0 | 1.0000 |
| kv2 | 12 | 0 | 1.0000 |
| realsense | 12 | 0 | 1.0000 |
| **xtion** | **60** | **12 (20%)** | **0.9950** |

For kv1/kv2/realsense every stored value is a multiple of 8, so the rotation
reduces to a divide-by-8 and per-image min-max preserves depth *ordering* — the
legacy input is wrong in scale but not in rank. **On `xtion` it is wrong in rank:**
20% of frames carry non-zero low 3 bits, affecting up to **32% of pixels** in the
worst sampled frame. Those bits are rotated to the top of the uint16, so an
affected pixel jumps by up to ~57 m and is then clipped to the 8 m ceiling;
reading the PNG directly leaves it at a mid-range value instead.

Two further divergences apply to every sensor: the **8 m clip is absent**, and
per-image min-max **destroys metric scale**, so an identical pixel value means a
different distance in different images.

**Requirement.** Decode to metres with the official rotation *before* forming the
student's input, and test the decoder against known encoded values.

### A3 — Labels mask padding only, so CE trains on the prompt

`dataset/datamodule/OneVision/CustomSUNRGBDOneVisionDataModule.py:145-146`

```python
labels = rgb_batch["input_ids"].clone()
labels[labels == self.processor.tokenizer.pad_token_id] = -100
```

Only pad tokens are ignored. Question tokens and image placeholders keep their
ids, so the causal loss is computed across the whole sequence — the model is
trained to reproduce the question as well as answer it. There is also **no causal
shift**: logits at position *t* predict token *t+1*, but the manual losses in §B
index `labels[t]` against the logits at *t*.

**Requirement.** CE and KD must target answer tokens at their causal prediction
positions, with prompt, image, and pad positions excluded.

---

## B. Losses

All four reproduced with `torch` on CPU using small deterministic tensors.

### B1 — LoCa writes to the wrong cells (advanced indexing, not scatter)

`.../phase1/OnlineKnowledgeDistillationLLavaOneVision.py:183-185`

```python
loca_teacher_probs[:, :, labels] = 1 - s * (...)
loca_teacher_probs[:, :, teacher_klogits] = s * non_target_probs_teacher
```

`labels` has shape `[B, L]`, so `x[:, :, labels]` selects a rank-**4** tensor of
shape `[B, L, B, L]`. Reproduced with `B=2, L=3, V=7`:

- the statement writes **6 of 7** vocabulary cells in *every* `(batch, position)`
  row — every label appearing anywhere in the batch, at every position;
- the intended per-position `scatter_` writes **1** cell per row;
- the two results are not equal.

The scalar arithmetic is correct — `sigma = 1/(1 - p_gt + p_wrong)` and
`p̃_g = 1 - s(1 - p_g)` both match the intended transformation — but the values
land in the wrong places, so the calibrated distribution is not the one the method
specifies, is not normalized, and does not preserve the gold class's rank.

**Requirement.** Use explicit `gather`/`scatter_` on valid answer positions, and
assert positivity, unit sum, gold-class rank, and preserved non-target ratios.

### B2 — The non-target class is chosen unconditionally as top-2

Lines 170-171 take `topk(2)[..., 1]`. When the gold class is *not* the teacher's
top-1, the highest wrong class **is** top-1, and the code selects the wrong one —
exactly the case LoCa exists to correct.

**Requirement.** Compute `max over c ≠ gold`, not "second highest".

### B3 — `reduction='mean'` under-scales KD by the vocabulary size

Line 191 uses `F.kl_div(..., reduction='mean')`, which divides by `B·L·V` rather
than summing over the vocabulary and averaging over valid positions. Reproduced at
`V=7`: `1.127e-01` against a correct `7.890e-01`, a factor of exactly **V**.

At the real vocabulary (~152k) the KD term is roughly **five orders of magnitude**
smaller than intended — effectively switched off beside the CE term. PyTorch emits
a warning here that `'mean'` does not match the KL definition and that
`'batchmean'` is intended.

This is *consistent with* the historical all-zero logit-KD tables noted in
`NEW_SUBMISSION.md` §4.1, and is a reason to re-execute those configurations
before concluding anything about ordinary KD. It does not by itself establish the
cause.

**Requirement.** Sum over vocabulary, average over valid answer positions, apply
`T²`, and compute the probability and KL operations in float32.

### B4 — Contrastive loss is identically zero at physical batch size 1

`contrastive_loss`, lines 410-414, builds an `[B, B]` similarity matrix and applies
`cross_entropy` against `arange(B)`. Reproduced:

| batch | loss |
|---:|---:|
| **1** | **0.000000** — no negatives exist; gradient exactly zero |
| 2 | 2.440781 |
| 4 | 3.664809 |

The pilot's default physical batch is one, so the contrastive term contributes
nothing. **Gradient accumulation does not help** — it enlarges the optimizer step,
not the candidate set.

**Requirement.** Use a cached negative bank (§8.3 of the plan: 255 negatives plus
one positive, sampled from distinct training scenes excluding room/sequence
neighbours), log the number of distinct scenes per candidate set, and make a
one-scene contrastive configuration a hard error.

### B5 — `gather` with `IGNORE_INDEX` raises

Lines 166-167 gather with `labels`, which §A3 shows contains `-100` whenever a
batch is padded. Reproduced: `index -100 is out of bounds for dimension 2 with
size 7` — a `RuntimeError`, not a silent wrong value.

**These findings compose.** LoCa can only have executed on batches with **no
padding**, i.e. physical batch size 1 — which is also precisely the regime where
B4 makes the contrastive term exactly zero. Any historical run using this path was
therefore single-scene, with a vocabulary-under-scaled KD term and a miswritten
LoCa target.

### B6 — Teacher vocabulary is sliced to student width

Line 155, `teacher_logits[:, :, :student_logits.size(2)]`, assumes the first *N*
vocabulary indices mean the same thing in both models. Safe only for a verified
identical tokenizer.

**Requirement.** Token-level KD only after tokenizer, special-token, answer-prefix
and position alignment are demonstrated — otherwise legal-answer candidate KD
(`experiment_protocol.md` §9.3).

---

## C. What this changes

1. **Do not port these losses.** Rebuild them as pure tensor functions with the
   §7.3 test list attached, per the plan's shared-runner design.
2. **Historical KD numbers need re-execution, not reinterpretation.** B3 and B4
   describe a configuration in which the KD and contrastive terms are close to
   inert. Those rows stay marked *historical, unverified* in the provenance CSV.
3. **The augmentation ablation cannot be reported at all** from this loader (A1).
4. **Any depth work must re-decode from raw**, and `xtion` frames must be included
   in the decoder test — a decoder validated only on kv1/kv2/realsense passes
   while still being wrong (A2).

## D. Gate G3 status

Replacement implementations landed 2026-09-05 as `distillation/losses.py` and
`distillation/depth_input.py`, with `tests/test_losses.py` (24 tests, the §7.3
mandatory list) and `tests/test_depth_input.py` (13 tests). Suite total: **140**.

| Requirement | Status |
|---|---|
| Depth decoding verified against known encoded values | ✅ `decode_raw_depth` checked against hand-computed values **and** byte-equal to the v2 generator decoder on real frames from all four sensors |
| Metric scale preserved across images | ✅ fixed 8 m ceiling; test recovers metres from the encoded channel |
| Labels masked exactly where the causal loss expects | ✅ `shift_for_causal_lm` + `valid_answer_mask`; padding provably cannot move a normalized loss |
| CE / KD / candidate-KD / feature / contrastive / LoCa unit-tested | ✅ 24 tests covering every §7.3 bullet |
| Temperature and reduction conventions verified | ✅ sums over vocabulary, averages over valid positions, scales by T²; regression test asserts it is not the V-divided convention |
| LoCa correctness | ✅ unit sum, non-negativity, gold strictly top-1 from any starting rank, margin exactly `1 − α`, non-target ratios preserved, `IGNORE_INDEX` never gathered |
| Contrastive rejects a candidate-free batch | ✅ empty bank raises instead of returning 0.0 |
| Contrastive negatives respect scene/split grouping | ✅ `negative_sampler.py` — dedupes by scene, excludes the anchor's `sequence_id` neighbours, seeded and reproducible, refuses a single-scene bank |
| X-Token: P-KL reduces to token KL under an identity mapping | ✅ asserted directly against `token_kd_loss` |
| X-Token: span alignment covers the whole answer, many-to-many | ✅ verified on the `"201"` vs `"2","0","1"` case and on agreeing tokenizations |
| X-Token: mapping is sparse and revision-keyed | ✅ two index tensors, never a dense matrix; a revision bump misses the cache |
| X-Token: top-K omitted mass is reported | ✅ grows monotonically as K shrinks |
| LoCa label dependence documented | ✅ `experiment_protocol.md` §8 records it as label-exposed |
| Augmentation flag affects the consumed tensor | ⬜ no augmentation in the new path yet (pilot runs with it off, §8.4) |
| One runner, ablations as configurations | ✅ `runner.py` — `RecipeConfig` refuses objectiveless recipes, LoCa composed with candidate/sequence KD, X-Token without `top_k`, and a CE term in stage F; the §9.2 matrix is a library, not forked code |
| Per-component loss logging | ✅ `compose_loss` returns each term separately — the log that would have surfaced the under-scaled KD term (B3) |
| Missing teacher signal fails loudly | ✅ a cache that did not supply the needed signal raises rather than silently skipping the term |
| Cache identity and reuse safety | ✅ `cache.py` — keys cover precision, top-K, both tokenizer revisions; a mismatched cache is refused **with the differing fields named** |
| Run manifest completeness | ✅ run ids hash the resolved config, so setting drift changes the id; X-Token runs must record both tokenizer revisions, mapping hash, and K |
| Genuine constrained decoding | ✅ `evaluation/candidate_scoring.py` — token trie makes an illegal answer unreachable; length/EOS policy declared explicitly |
| Depth-only inference harness | ✅ `inference_isolation.py` — RGB-path-mutation invariance, permission denial, and an open() tracer. Harness tested; **running it needs a checkpoint** |
| Stage freezing, optimizer groups, checkpoint lineage | ⬜ pending a real model |
| Tiny-subset overfit check | ⬜ pending GPU |
| Depth-only inference *executed* on a trained student | ⬜ pending GPU |

**G3's remaining rows all need a model.** Every part that could be verified
without one now is: the objective surface, the data path, the cache contract, the
decoding mechanism, and the isolation harness. The legacy path is superseded, not
patched.
