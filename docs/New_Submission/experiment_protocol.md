# Experiment protocol — RGB-to-depth distillation for depth-only VQA

**Status:** LOCKED research contract (Phase 1 of `docs/NEW_SUBMISSION.md` §19)
**Created:** 2026-09-05
**Author:** Shayekh Mohiuddin Ahmed Navid
**Supersedes:** nothing. This is the first version.
**Governs:** every training run, cache, checkpoint, prediction file, and reported
number in the resubmission.

This document exists to be written *before* results are seen. Its purpose is to
remove the degrees of freedom that made the previous submission unfalsifiable:
which comparison counts, which number is the headline, how many seeds, what
counts as a win, and what supervision each row was allowed to use.

Nothing in this document may be revised in response to a test-set result. All
changes require a dated entry in §13.

---

## 1. Primary hypothesis

**H1.** For a small vision-language model deployed with depth as its only visual
input, supervision transferred from a larger RGB teacher improves five-type test
macro accuracy over **matched depth-only cross-entropy training** on the same
data, labels, schedule, and model-selection rule.

The direction is fixed and one-way: **RGB teacher during training → depth-only
student at inference.** No claim is made about the reverse direction, about
modalities other than depth, or about sensory substitution in general.

**Null hypothesis (H0).** The paired difference (KD − matched CE) in five-type
test macro accuracy is zero.

H1 is a claim about *value added beyond supervision*. A KD result that only beats
a zero-shot or an under-trained CE model does not support H1 and must not be
reported as if it does.

---

## 2. Primary endpoint

**Five-type test macro accuracy under canonicalized greedy exact-match scoring**,
on the frozen v2.4 test split, computed as:

```
A_k      = (1 / N_k) · Σ_{i ∈ type k} 1[ canon(â_i) = canon(a_i) ]
A_macro  = (1 / 5) · Σ_{k=1..5} A_k
```

where `k` ranges over the five released question types, unweighted. This is a
single primary endpoint. Every other metric named in this protocol is secondary
or diagnostic and may not replace it after results are seen.

**Decoding for the primary endpoint** is greedy (`do_sample=False`,
`max_new_tokens=16`, standard EOS termination), using the frozen prompt in §8.3.
Constrained-decoding accuracy is reported as a **separate diagnostic column**,
never as the primary number.

**Unanswered, invalid, missing, and unparsable predictions count as wrong** and
remain in the denominator. Invalid-output rate is reported alongside, separately.

### 2.1 Secondary metrics (declared now, confirmatory family)

| Metric | Purpose |
|---|---|
| Per-type accuracy (all five, always shown) | Prevents a macro average from hiding a per-type regression |
| Depth-relation aggregate over `identify_superlative`, `relative_depth`, `nearest_object` | Tests measured-depth reasoning specifically |
| Micro accuracy | Descriptive; different denominator from macro |
| Macro-F1 over fixed legal label sets (binary types; first/second-mentioned for `relative_depth`) | Guards against majority-class inflation |
| Invalid-output rate | Separates visual errors from formatting errors |

Holm correction applies across the declared confirmatory family. Any slice not
listed above is **exploratory** and must be labeled as such in the manuscript.

---

## 3. Benchmark definition (locked)

**Dataset:** VQA-SUNRGBD **v2.4**, frozen, used unchanged.

| Identity | Value |
|---|---|
| Freeze manifest | `release/VQA-SUNRGBD-v2/FROZEN_v2.4.json` |
| Manifest sha256 | `659d65f83548f9554af46061b3fdb03287bae7ff7294716d3cb2549dc636f8d2` |
| Frozen at | 2026-09-05T01:10:06Z |
| `train.csv` sha256 | `acbb5eab2a028c792fdc09ff7f2c86d01177bdef937d28668c6e1a31618d1bfa` (15,278 rows) |
| `val.csv` sha256 | `a3d649ca83fbd7c9bbab8c553b7cb2c7d33decfc663a50bdb4e92dec99807883` (1,720 rows) |
| `test.csv` sha256 | `2b2db9dba8237ea8e8561f530e76ae3819145e6cc9d5e56c5f504919c1fedc72` (12,463 rows) |
| Verification | `freeze_release.py --verify v2.4` → "verified, no drift" (2026-09-05) |

### 3.1 The five fixed question types

Counts below were recomputed directly from the released CSVs on 2026-09-05 and
match `NEW_SUBMISSION.md` §2.3.

| Type | Train | Val | Test | Answer space |
|---|---:|---:|---:|---|
| `existence` | 4,186 | 352 | 2,558 | binary (yes / no) |
| `identify_superlative` | 2,615 | 341 | 2,490 | open vocabulary (object name) |
| `left_right` | 2,995 | 340 | 2,470 | binary (left / right) |
| `nearest_object` | 2,758 | 343 | 2,474 | open vocabulary (object name) |
| `relative_depth` | 2,724 | 344 | 2,471 | item-specific pair (two named objects) |
| **Total** | **15,278** | **1,720** | **12,463** | |

**The macro average is the unweighted mean of these five per-type accuracies.**
Types are never dropped, merged, reweighted, or added. No `v2.5` release is
created for this submission.

### 3.2 Prohibited dataset actions

Rebuilding the release; moving rows between splits; substituting or repairing
rows; adding or removing a question type; restoring counting, color, `largest`,
or scene classification; writing into `release/VQA-SUNRGBD-v2/`.

**Audit artifacts (`audit/`) are out of scope.** They are not cited, reported,
or referenced in the paper or supplement, and no human-verification,
human-audited, gold-standard, or error-rate claim is made about the labels.

---

## 4. Primary comparator

The primary comparator is **the stronger of B3 and B4, selected on validation
macro accuracy before the confirmatory test evaluation** — that is, the best
matched depth-only CE student, not a zero-shot model and not a deliberately
weakened baseline.

| Row | Recipe |
|---|---|
| B3 | Depth CE, jointly tuning the permitted vision/projector/language modules |
| B4 | Depth CE in **both stages**, using the proposed stage schedule and trainable-module masks |

B4 exists specifically so that extra optimization and staging cannot be credited
to distillation.

**Fairness obligations toward the comparator** (all binding):

- Identical train/val/test question IDs, prompt, canonicalization, decoding,
  depth encoding, resolution, and augmentation setting.
- Identical initial student checkpoint and identical seed set.
- Identical total data exposure and optimizer-step accounting.
- **Equal hyperparameter-search budget**: the same number of tuning trials over
  the same learning-rate grid (§7.2). CE is tuned first and fully.
- Identical trainable-module masks. KD may not adapt a module that CE is
  forbidden to adapt.
- Checkpoints selected by validation macro accuracy under the common decoding
  protocol — never by comparing differently scaled CE and KD losses.

GPU-hours are additionally reported for both, because teacher supervision costs
extra computation. Exposure-matched and compute-matched comparisons are reported
as distinct rows and are never conflated.

---

## 5. Seeds, selection, and uncertainty

| Item | Locked value |
|---|---|
| Seeds (confirmatory) | **17, 42, 2026** — all three reported individually, plus mean and sample SD |
| Validation selection | Highest **validation five-type macro accuracy** under the locked decoding protocol; ties broken by earlier step |
| Test evaluation | **Once**, on the selected checkpoint, after settings are locked. No re-selection after seeing test output |
| Uncertainty | **Paired cluster bootstrap**, 10,000 replicates, bootstrap seed **20260905** |
| Cluster unit | Scene / room group (`sequence_id`); all questions of a sampled group travel together |
| Aggregation inside bootstrap | The **full macro metric is recomputed inside each draw** — never averaged from per-question intervals |
| Seed variation | Reported **separately** from evaluation-set uncertainty; a test bootstrap alone does not estimate retraining variability |
| Joint estimate | Hierarchical bootstrap over seeds × scene groups, explicitly labeled; interpreted cautiously with only three seeds |
| Multiplicity | Holm correction across the §2.1 confirmatory family |
| Reported quantity | **Confidence interval on the paired difference**, not two separate per-model intervals |

Scene-aware uncertainty applies equally to the paraphrase, corruption, and
external-transfer comparisons.

### 5.1 Success threshold (planning target, not a scientific law)

A **≥ 2 percentage-point** gain in five-type test macro accuracy over the
strongest matched CE control, with a **paired 95% interval excluding zero**,
is treated as evidence that the method merits the full study.

Smaller repeatable gains may still matter and must be reported honestly. Larger
gains are **not** accepted if traceable to unequal supervision, unequal tuning
budget, invalid labels, or leakage.

**Prohibited:** running additional seeds, benchmark variants, or subsets until an
interval excludes zero. The expansion policy in §7.3 is the only permitted route
to more data.

---

## 6. Inference contract (the deployable student)

At inference the student receives **exactly two things: the depth image and the
question string.**

**Denied at inference**, without exception:

RGB pixels or paths · teacher model, logits, features, or cache · object masks ·
3-D boxes · scene-type or annotation metadata · candidate object lists derived
from annotations · the gold answer · any target label · answer-space columns for
open-vocabulary types.

### 6.1 How this is verified (Phase 4 gate, §19 line 807)

1. Export the student with no teacher weights or cache readers on the inference path.
2. Run a full evaluation smoke pass with RGB directories and teacher-cache
   directories made **inaccessible at the filesystem level**.
3. Mutate RGB paths while holding depth and questions fixed — predictions must be
   **bitwise identical**.
4. Trace file access (or instrument the loader) to confirm only authorized depth
   inputs are opened.
5. Record whether the supplied depth is registered, inpainted, or RGB-completed.
   Any raw-sensor deployment claim requires a separate raw-depth evaluation.

### 6.2 Honest description of the depth input

v2.4 depth is benchmark depth. Robustness of the model *to that representation*
is not robustness of a full acquisition pipeline, and the manuscript must say so.

---

## 7. Run budget, early stopping, and drop criteria

### 7.1 Budget structure

Compute is currently constrained (§9). The budget is therefore expressed in
**configuration-seed results**, not GPU-hours, and is re-estimated after the
Phase 5 pilot produces measured throughput.

| Bucket | Configuration-seeds | Gate |
|---|---:|---|
| Pilot (one seed): B3, B4, D3, D5, D6 | 5 | Runs on 16 GB now |
| Confirmatory core: B3–B6, D1–D9 × 3 seeds | 39 | Needs locked teacher precision |
| Cross-model replication | 9 | After primary recipe is selected |
| Screens (SmolVLM2, Florence-2, Gemma), one seed | 3 | Promote at most **one** compact student |
| Required extra controls (question-only, corruption-augmented RGB) × 3 | 6 | |

This is a planning envelope, not a promise. D0 checkpoints are reused, not
retrained; shared stage-one checkpoints are branched across compatible S2 rows.

### 7.2 Tuning budget

Learning rate grid: **5e-6, 1e-5, 2e-5** (three trials). CE is tuned first across
all three. KD then receives a **comparable declared trial count** with fixed
preprocessing. The Cartesian product of the §8.4 pilot table is explicitly **not**
run. Failed trials are archived with the reason for stopping.

### 7.3 Early stop and drop criteria

- **Stage-two stop:** up to 5 epochs; stop when validation macro accuracy fails to
  improve over 2 consecutive evaluation points. Every compared row gets **equal
  checkpoint-evaluation opportunities**.
- **Numerical failure** (loss NaN/inf, 0% task accuracy, gradient explosion) is an
  implementation defect to diagnose — **never** reported as a research finding
  about the method. Cf. the historical all-zero KD tables (`NEW_SUBMISSION.md` §4.1).
- **Drop an optional experiment** when: its compatibility gate (§8.1.1) fails; its
  CE baseline is not usable; or its measured cost exceeds the remaining budget.
  Dropping is recorded in §13 with the reason — silent omission is prohibited.
- **Scope reduction under pressure** is transparent: keep three seeds for the
  primary CE-vs-KD claim, demote secondary ablations to one seed **labeled
  exploratory**, and delete the claims they no longer support.

---

## 8. Permitted training signals per experiment row

This table is the **label-access inventory** demanded by Reviewer #5. A row may
use only what its line permits. "Gold prefix" means gold answer tokens supplied as
teacher-forcing context during distillation.

| ID | Recipe | Student visual input | Gold answers (CE) | Gold in LoCa | Gold prefix | Teacher logits / candidates | Teacher features |
|---|---|---|---|---|---|---|---|
| B0 | Random / train-majority / TF-IDF question-only | none | priors only | – | – | – | – |
| B1 | Pretrained student, zero-shot | depth | – | – | – | – | – |
| B2 | Pretrained student, zero-shot | RGB | – | – | – | – | – |
| B3 | Depth CE (joint modules) | depth | ✅ | – | – | – | – |
| B4 | Depth CE, stage-matched | depth | ✅ | – | – | – | – |
| B5 | RGB CE, matched student | RGB | ✅ | – | – | – | – |
| B6 | RGB-D supervised reference | RGB+depth | ✅ | – | – | – | – |
| D0 | F checkpoint before S2 | depth | – | – | – | – | ✅ |
| D1 | S2: CE + raw KD | depth | ✅ | – | ✅ | ✅ | – |
| D2 | S2: CE + LoCa KD | depth | ✅ | ✅ | ✅ | ✅ | – |
| D3 | F → S2: CE only | depth | ✅ (S2) | – | – | – | ✅ (F) |
| D4 | F → S2: CE + raw KD | depth | ✅ | – | ✅ | ✅ | ✅ |
| D5 | F → S2: CE + LoCa KD | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
| D6 | P → S2: CE + LoCa KD *(intended recipe)* | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
| D7 | Joint feature + CE + LoCa KD | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
| D8 | Cosine/MSE feature → S2: CE + LoCa KD | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
| D9 | F → S2: teacher-prefix raw KD only | depth | ❌ | ❌ | ❌ (teacher-generated) | ✅ | ✅ |

### 8.1 D9 label-access rule (strict)

Removing CE **does not** make a run label-free. D9 is the only label-restricted
row, and it qualifies only if **all** of the following hold:

- No gold-based LoCa.
- No gold answer prefixes — prefixes are **teacher-generated**.
- No teacher fine-tuned on the target answers.
- No correctness-based filtering of training items.
- **Gold answer columns are removed before the training/cache interface**, and the
  run succeeds with them absent.

Gold labels remain available to the *evaluator* only. If labeled validation data
select the D9 checkpoint, it is called **"training-answer-free adaptation with
labeled model selection"** — never "zero-shot" and never "label-free". Benchmark
construction itself used annotations, so the resource as a whole is not
annotation-free.

### 8.2 Stage definitions

- **F** — stage one, feature alignment only; selected vision parameters update;
  projector and language parameters frozen.
- **P** — stage one as submitted: feature alignment **plus** a small raw KD term.
  If that KD uses gold answer prefixes, **P is label-exposed even without CE**.
- **S2** — stage two; vision encoder frozen; declared projector/language
  parameters train.
- **Raw KD** — teacher→student KL without gold-conditioned correction.
- **LoCa KD** — label-conditioned calibration (ECAI 2024), then KL. Attributed as
  existing work, not presented as a new calibration method.

### 8.3 Frozen prompt and decoding contract

Identical for every model, embedded in each model's **native** chat template:

```
Answer with only the short answer: yes, no, left, right, or the object name,
as appropriate. Use no explanation.
```

This **replaces** the legacy "single word or number" instruction, which conflicts
with legitimate multiword answers such as "tissue box".

`do_sample=False` · `max_new_tokens=16` · standard EOS · no repetition penalty ·
no n-gram restriction · no sampling temperature argument where the API ignores or
rejects it under greedy decoding.

Prompt text, prompt ID, processor revision, canonicalizer, and evaluator hashes
are stored beside **every** prediction file.

---

## 9. Compute plan: 16 GB now, 24 GB later

### 9.1 Current reality

`NEW_SUBMISSION.md` was written for an RTX 4090 (24 GB). The available machine is
an **RTX 4080 SUPER, 16,376 MiB (≈16 GB)**; the 4090 is temporarily unavailable
and expected back. This protocol therefore splits work into what runs now and
what waits.

**The portfolio survives the reduction.** Every *student* is ≤ 2.2 B parameters
and trains comfortably in 16 GB with LoRA. Only the *teachers* are large — and
under `NEW_SUBMISSION.md` §8.2 the teacher **never coexists with student
training**: it runs alone, its signals are cached to disk, and it is unloaded.
Teacher size is therefore bounded by inference memory, not training memory.

### 9.2 Verified portfolio (checked on HuggingFace, 2026-09-05)

All checkpoints resolve and are genuinely multimodal (`image-text-to-text`).

| Role | Checkpoint | Params | Arch |
|---|---|---:|---|
| Primary teacher | `Qwen/Qwen3.5-9B` | 9.65 B | `Qwen3_5ForConditionalGeneration` |
| Primary student | `Qwen/Qwen3.5-0.8B` | 0.87 B | `Qwen3_5ForConditionalGeneration` |
| Replication-A student | `OpenGVLab/InternVL3_5-1B` | 1.06 B | `InternVLChatModel` |
| Replication-B teacher | `OpenGVLab/InternVL3_5-8B` | 8.53 B | `InternVLChatModel` |
| Screen student | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 0.51 B | `SmolVLMForConditionalGeneration` |
| Screen student | `microsoft/Florence-2-large` | 0.78 B | `Florence2ForConditionalGeneration` |
| Screen teacher | `google/gemma-4-12B-it` | 11.96 B | `Gemma4UnifiedForConditionalGeneration` |
| Deferred | `Gemma 4 12B → MiniCPM-V-4.6` | — | Not started for this submission |

Note: Qwen3.5 is natively multimodal — there is no separate `-VL` suffix, and
`Qwen/Qwen3.5-9B` / `-0.8B` are the correct vision-capable checkpoints.

Exact commit SHAs, processor revisions, chat templates, image-resolution policy,
and licenses are pinned during the Phase 5 compatibility gate and recorded per run.

### 9.3 KD contract per pair

**Tokenizer compatibility is not a model-selection criterion.** The distillation
layer bridges tokenizer differences (`NEW_SUBMISSION.md` §7.2.1). Teachers are
chosen for capability, students for deployment constraints. No pair in this
portfolio is selected, excluded, or downgraded because of its tokenizer.

| Pair | KD objective |
|---|---|
| `Qwen3.5-9B → Qwen3.5-0.8B` | **X-Token P-KL** as the primary soft objective. Its identical tokenizer additionally permits aligned token-level logit KD, used as an **exactness check**: under an identity mapping the two must agree, and disagreement is a projection bug. Candidate-answer KD retained as the common reference. |
| `Qwen3.5-9B → InternVL3_5-1B` | X-Token P-KL; candidate-answer KD as reference (+ optional pooled-feature projector after smoke test) |
| `InternVL3_5-8B → Qwen3.5-0.8B` | X-Token P-KL; candidate-answer KD as reference (+ optional pooled-feature projector after smoke test) |
| `Qwen3.5-9B → SmolVLM2-500M` | X-Token P-KL; candidate-answer KD as reference |
| `Qwen3.5-9B → Florence-2-large` | Candidate-answer KD; X-Token only if its encoder-decoder output exposes a well-defined assistant answer span. `trust_remote_code=True` — pin model **and** code revision, review the code before executing |
| `gemma-4-12B-it → Qwen3.5-0.8B` | X-Token P-KL; candidate-answer KD as reference. The portfolio's clearest unrelated-tokenizer pair, and therefore the most informative screen for whether X-Token generalizes beyond related families. Feasibility screen, never a headline result |

**Three distillation modes** are in scope and compared under matched conditions:
sequence-level (teacher text only, the fallback when logits are unavailable),
X-Token cross-tokenizer P-KL, and the hybrid
`L_total = λ_CE·L_CE + λ_KD·L_XToken`. The X0–X5 rows in `NEW_SUBMISSION.md` §9.2
isolate the mode while holding the recipe fixed — X-Token's place in the method is
to be earned against the simpler modes, not assumed.

**Fixed before the confirmatory runs:** top-K teacher logits (K ∈ {2048, 4096,
8192}, starting at 4096), the within-span aggregation rule, and λ_CE/λ_KD. K and
the tokenizer revisions are part of the frozen target source — changing either
invalidates the cache exactly as a precision change does (§9.5), and makes two
rows incomparable.

Cross-family pairs define teacher and student distributions over the **same
finite legal answer set**, each scored with its own tokenizer and prompt, then
normalized. Logits from unrelated tokenizers are never padded, truncated, or
index-matched.

### 9.4 VRAM estimates (to be replaced by Phase 5 measurements)

Weight-only arithmetic; activations, vision tokens, and optimizer state are extra.
These are **estimates, not measurements**.

| Workload | bf16 | int8 | NF4 | Fits 16 GB? |
|---|---:|---:|---:|---|
| Qwen3.5-9B teacher inference | ~19.3 GB | ~9.7 GB | ~5.4 GB | **bf16 no** → quantize |
| InternVL3_5-8B teacher inference | ~17.1 GB | ~8.5 GB | ~4.8 GB | **bf16 no** → quantize |
| gemma-4-12B teacher inference | ~23.9 GB | ~12.0 GB | ~6.7 GB | **bf16 no** → NF4 |
| Qwen3.5-0.8B student, LoRA r16 | ~1.7 GB + states | — | — | **yes** |
| InternVL3_5-1B student, LoRA r16 | ~2.1 GB + states | — | — | **yes** |
| SmolVLM2-500M / Florence-2-large student | ~1.0–1.6 GB + states | — | — | **yes** |

Target ≈ 2 GB headroom rather than configuring at the device limit.

### 9.5 Teacher-precision decision — OPEN, requires author sign-off

This is the one place where 16 GB genuinely changes the science.

Cache keys include precision (`NEW_SUBMISSION.md` §8.2), and all compared KD rows
must draw from **one frozen target source** — precision may not be mixed across
rows. Caching at NF4 now and re-caching at bf16 on the 4090 produces two
incompatible sources.

| Option | Description | Consequence |
|---|---|---|
| **A (recommended)** | Use 16 GB for Phases 1–5 only: evaluator repair, correctness tests, compatibility gates, throughput profiling, 32-example overfit, and the **one-seed pilot** (B3, B4, D3, D5, D6) with an NF4 teacher. All results labeled **PILOT**. Run confirmatory caching + three-seed Phases 6–8 on the 4090 at a single locked precision. | Costs nothing: Phases 1–4 are CPU/writing work and Phase 5 is explicitly a profiling pilot. Matches the plan's own ordering. |
| **B** | Lock **NF4 (or int8) as the frozen target source for every row**, including confirmatory, and never switch. | Unblocks the full study on 16 GB. Requires a documented bf16-vs-quantized teacher agreement check on dev examples, and a stated manuscript limitation. |
| **C** | Wait for the 4090 before any teacher caching. | Wastes the current window. Not recommended. |

**DECIDED 2026-09-05: Option A.** Confirmed by the author. Pilot results produced
on the 16 GB card are labeled PILOT and are never promoted into a confirmatory
table. Confirmatory teacher caching and all three-seed runs wait for the 4090 and
use one locked precision.

Binding consequences of this choice:

- Every artifact produced before the 4090 returns carries `PILOT` in its run
  manifest, and no PILOT number enters a main or ablation table.
- Phases 1–4 and the Phase 5 profiling pilot proceed now at full effort; they are
  CPU- or small-GPU-bound and lose nothing to the smaller card.
- The NF4 teacher used for pilot caching is a *pilot* target source. Its caches
  are discarded, not reused, when confirmatory caching begins — cache keys
  include precision, so they would not silently mix in any case.
- Gate G4 is evaluated twice: once against 16 GB for the pilot, once against the
  4090 for the confirmatory configuration.

If the 4090 does not return before Phase 6 is otherwise ready, switching to
Option B requires a dated entry in §13 — not silent continuation.

---

## 10. Run ID format and artifact carriage

**Format:**

```
{YYYYMMDD}-{pair}-{recipe}-s{seed}-{cfg8}
```

| Field | Meaning | Example |
|---|---|---|
| `YYYYMMDD` | Run start date (UTC) | `20260907` |
| `pair` | Teacher→student slug, or `none` for non-distilled rows | `qw9b2qw08b`, `none2qw08b` |
| `recipe` | Row ID from §8 | `B3`, `D6` |
| `seed` | Training seed | `s17` |
| `cfg8` | First 8 hex of sha256 of the **fully resolved** `configuration.yaml` | `3f9a1c02` |

Example: `20260907-qw9b2qw08b-D6-s17-3f9a1c02`

The `cfg8` component makes silent setting drift detectable: two runs claiming the
same recipe with different resolved configs cannot share a run ID.

**Every** result, cache shard, checkpoint, log, prediction file, and table cell
carries its run ID. Directory contract per `NEW_SUBMISSION.md` §15:

```
runs/resubmission/<run_id>/
  manifest.json          # data/code/model/prompt hashes; seed; precision; modules
  configuration.yaml     # fully resolved settings, every loss weight
  parent_checkpoint.json # explicit stage lineage + checkpoint hash
  training_metrics.csv
  validation_predictions.csv
  test_predictions.csv
  metrics.json           # evaluator version, denominators, per-type + aggregates
  resource_usage.json    # GPU, peak VRAM, runtime, teacher/cache cost
```

Prediction files carry `question_id`, `prediction` (raw), `prediction_canonical`,
`run_id`, `failure_status`. **`question_id` is the frozen release ID**, never a
batch index. The v1-schema adapter's `IDs` column was verified on 2026-09-05 to
equal the release `question_id` exactly — same set, same order, zero duplicates,
across all three splits.

**Teacher cache keys** additionally include: scene ID, dataset version,
question/prompt hash, prefix source, model + processor revision, depth/RGB
transform, **precision**, **top-K**, **both tokenizer revisions**, feature layer,
and crop aggregation. A change to any of these invalidates the affected entries.

**X-Token runs additionally record** in `manifest.json`: the distillation mode
(sequence / candidate / X-Token / hybrid), the teacher and student tokenizer
revisions, the sparse mapping's content hash, its match coverage (exact textual
matches against entries resolved by decode-and-retokenize), K, and the mean
omitted teacher probability mass. The vocabulary mapping is cached separately
from the per-scene signals — it depends only on the tokenizer pair, so it is built
once per revision pair and reused across every run of that pair.

---

## 11. Gates this protocol is subject to

| Gate | Condition | Status |
|---|---|---|
| G1 dataset | v2.4 hash verification passes; five-type contract frozen; construction claims trace to release artifacts | ✅ verification passed 2026-09-05; `dataset_protocol.md` outstanding |
| G2 evaluator | Edge cases verified; random baseline fixed; ID integrity locked | ❌ **open** — defect confirmed at `evaluate.py:189` |
| G3 training | Tested losses, correct masks/alignment, valid contrastive candidates, depth-only inference | ❌ open |
| G4 resources | Pilot + resume fit the actual GPU with measured throughput | ❌ open — 16 GB, see §9.5 |
| G5 central benefit | Three-seed KD gain over strongest matched CE with paired uncertainty | ❌ open |
| G6 component rationale | Each retained component beats its simpler alternative | ❌ open |
| G7 generalization | Independent scenes + second-family evidence support the stated scope | ❌ open |
| G8 submission readiness | Every major claim has a verified artifact | ❌ open |

### 11.1 Known blocking defect (Phase 3)

`evaluate.py:189` computes `str(row["answer_space"] or "")`. For open-vocabulary
types `answer_space` is `NaN`, and **`NaN` is truthy in Python**, so the `or ""`
fallback never fires. `str(nan)` → `"nan"` → `options == ["nan"]`, and the
`open_vocabulary` fallback on line 191 is unreachable. Every open-vocabulary row
"predicts" the literal string `nan`.

Observed consequence in the current baseline output: `identify_superlative`
random = **0.0%**, `nearest_object` random = **0.0%**. Correct NaN handling
already exists nine lines earlier in `snap_to_answer_space` (line 105).

**Fixing this will move the published 30.2% random macro figure.** No comparative
result may be reported until G2 closes.

---

## 12. Outcome rules (decided in advance)

| Observed outcome | Required claim change |
|---|---|
| KD beats only zero-shot, not matched CE | Claim adaptation to depth. **Do not** claim distillation beats supervision |
| CE matches the full pipeline | Favor CE operationally; pivot to a benchmark/analysis paper |
| Raw KD matches LoCa | Remove LoCa from the central method claim |
| MSE/cosine matches contrastive transfer | Use the simpler loss; report it |
| Teacher-free soft labels match LoCa | Describe label regularization, not teacher-knowledge transfer |
| Blank/shuffled depth scores similarly | Investigate priors and leakage before any depth-grounding claim |
| External gains disappear | Report benchmark dependence. **Do not hide the external result** |
| Only synthetic low-light evidence exists | Limit the claim to simulated RGB degradation |
| Gains confined to object identity | Describe semantic adaptation, not improved metric-depth reasoning |

Negative results and failed components that materially limit the final claim are
preserved in the manuscript.

---

## 13. Amendment log

Every change to §§1–12 requires a dated row here. Amendments made **after** any
confirmatory test result is observed must state what was already seen.

| Date | Section | Change | Reason | Test results seen at time of change? |
|---|---|---|---|---|
| 2026-09-05 | — | Initial version | Phase 1 of `NEW_SUBMISSION.md` §19 | No |
| 2026-09-05 | §9.5 | Teacher precision resolved to **Option A** | Author decision; 4090 temporarily unavailable but expected back | No |
| 2026-09-05 | §9.3, §10 | Cross-tokenizer distillation (X-Token) adopted. Soft KD is X-Token P-KL for every pair; candidate KD demoted to common reference; token-level KD demoted to an exactness check on the identical-tokenizer pair. Tokenizer compatibility removed as a model-selection criterion. Sequence-level and hybrid modes added, with X0–X5 rows to earn the objective against them. Manifests must record distillation mode, both tokenizer revisions, mapping hash, and K | Author decision, specified in `NEW_SUBMISSION.md` §7.2.1. Teacher choice should follow capability and student choice deployment constraints; tokenizer compatibility should not restrict the experimental grid | No |

### 13.1 Open decisions awaiting author sign-off

1. ~~**§9.5 teacher precision**~~ — **RESOLVED 2026-09-05: Option A.**
2. ~~**Expected 4090 return date**~~ — **expected 2026-09-06 or 2026-09-07**
   (author, 2026-09-05). Option B is therefore not needed: the confirmatory stage
   is days away, not indefinite, so pilot caches stay pilot and confirmatory
   caching waits for the 24 GB card at a single locked precision, exactly as
   Option A specifies. Revisit only if the date slips materially.
3. **Target venue** — deliberately unset; `NEW_SUBMISSION.md` §17 records that
   BMVC 2026 and WACV 2027 deadlines have passed. Choose after Gate G5.
4. **Whether B6 (RGB-D reference) is retained** — it needs a two-image VLM input
   path and is a contextual baseline, not a matched competitor.

---

## 14. Phase 1 checklist status

Against `docs/NEW_SUBMISSION.md` §19 Phase 1:

- [x] Create the experiment protocol before running a new training job → this file
- [x] State the primary hypothesis → §1
- [x] Name the primary endpoint → §2
- [x] List the five fixed types and the macro definition → §3.1
- [x] Predeclare the primary comparator → §4
- [x] Predeclare seeds, validation selection, test rule, paired uncertainty → §5
- [x] Define run budget, early-stop rule, drop criteria → §7
- [x] Record the target inference contract → §6
- [x] Record permitted training signals per row → §8
- [x] Record the selected model portfolio → §9.2 (all checkpoints verified to exist)
- [x] Specify the soft-KD objective per pair, and record that tokenizer compatibility is not a selection criterion → §9.3 (amended 2026-09-05)
- [x] Assign a unique run ID format and require it on every artifact → §10

**Phase 1 is complete pending author sign-off on §13.1.**

Next: Phase 2 (`dataset_protocol.md`, closing G1) and Phase 3 (evaluator repair,
closing G2). Phase 3 is the critical path — no comparative number is trustworthy
until the defect in §11.1 is fixed.
