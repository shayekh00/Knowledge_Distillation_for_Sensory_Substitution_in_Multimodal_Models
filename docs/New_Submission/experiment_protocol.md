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
| Seeds (confirmatory) | **17 only** — single seed for every row (author decision 2026-09-07, §13). Was 17/42/2026; see that amendment for what this costs and the limitation it obliges |
| Validation selection | Highest **validation five-type macro accuracy** under the locked decoding protocol; ties broken by earlier step |
| Test evaluation | **Once**, on the selected checkpoint, after settings are locked. No re-selection after seeing test output |
| Uncertainty | **Paired cluster bootstrap**, 10,000 replicates, bootstrap seed **20260905** |
| Cluster unit | Scene / room group (`sequence_id`); all questions of a sampled group travel together |
| Aggregation inside bootstrap | The **full macro metric is recomputed inside each draw** — never averaged from per-question intervals |
| Seed variation | **Not estimated** under the single-seed policy. The paired cluster bootstrap covers evaluation-set uncertainty only — it does not estimate retraining variability, and must never be presented as though it does. This is a **declared manuscript limitation**, not an omission |
| Joint estimate | **Withdrawn** with the single-seed policy — a hierarchical bootstrap over seeds × scene groups needs more than one seed. Scene-group bootstrap only |
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
| Confirmatory core: B3–B6, D1–D9 × 1 seed (§5, 2026-09-07) | 13 | Needs locked teacher precision |
| Cross-model replication | 9 → **recount pending** | After primary recipe is selected; composition of this 9 is not stated here, so its seed multiplier is unknown — recount before relying on it |
| Screens (SmolVLM2, Florence-2, Gemma), one seed | 3 | Promote at most **one** compact student |
| Required extra controls (question-only, corruption-augmented RGB) × 1 seed | 2 | |

This is a planning envelope, not a promise. D0 checkpoints are reused, not
retrained; shared stage-one checkpoints are branched across compatible S2 rows.

**Re-estimated from measured throughput, 2026-09-06** (§7.1 asks for this once
Phase 5 profiling exists; it now does — §13.1 point 2, Gate G4):

- A single-stage row (B3/B4/B5-shaped: one CE or CE+KD training pass, batch-4,
  no gradient checkpointing, plus one val-split eval) measures **~24.3 minutes**
  end to end (21.1 min train + 3.2 min eval; `pilot_findings.md` §10.2).
- A two-stage row (F→S2 or P→S2: D3–D9) pays that cost roughly twice, since
  stage one is a separate training pass before S2 begins — **~45–50 minutes**,
  before accounting for D0's shared/reused checkpoints reducing the count of
  stage-one passes actually needed.
- Teacher cache generation is a one-time cost, not per-row: **~34 minutes** for
  the full 15,278-row train split at the measured 7.53 ex/s (§13.1 point 2),
  amortized across every KD row that reads from it.
**Superseded 2026-09-07 — the two bullets above measured *one-epoch* rows.**
Under §7.3's 10-epoch/patience-2 policy a row trains for as many epochs as its
own validation curve earns, and the measured cost rose by roughly 5–9x:

| Row (seed 17, measured) | Epochs run | Train | + eval |
|---|---:|---:|---:|
| B3 depth CE | 5 | 127.6 min | ~131 min |
| B3 depth CE (seed 42) | 5 | 123.0 min | ~126 min |
| X2 depth CE+X-Token KD | 6 | 149.8 min | ~153 min |
| B5 rgb CE | 9 | 216.9 min | ~220 min |

- A single-stage row now measures **~2.1–3.7 hours** end to end, against the
  ~24.3 minutes estimated from one-epoch profiling. KD's own per-step overhead
  remains small (10.2 vs 12.3 ex/s, ~17%); the epoch count dominates it.
- A two-stage row (F→S2, D3–D9) pays that roughly twice: **~4–7 hours**.
- At these rates the revised **32 configuration-seeds** (pilot 5 + confirmatory
  core 13 + cross-model 9 + screens 3 + controls 2) come to roughly
  **80–110 GPU-hours** sequential — about **3.5–4.5 days** of continuous
  compute, not counting the one-time cache.
- This re-estimate is the direct reason the seed set was cut to one (§5,
  2026-09-07): at three seeds the same ladder would have been 10–14 days of
  uninterrupted single-card compute, which was never affordable alongside the
  still-unbuilt LoCa and feature-alignment stages. This is a planning number,
  not a promise
  (expected small; not yet measured — the cached-teacher KD loss reads a
  top-K array off disk rather than running a live teacher forward pass, so the
  dominant cost stays the student's own forward+backward, same as CE).

### 7.2 Tuning budget

Learning rate grid: **5e-6, 1e-5, 2e-5** (three trials). CE is tuned first across
all three. KD then receives a **comparable declared trial count** with fixed
preprocessing. The Cartesian product of the §8.4 pilot table is explicitly **not**
run. Failed trials are archived with the reason for stopping.

### 7.3 Early stop and drop criteria

- **Stage-two stop:** up to **10 epochs** (raised from 5, author decision,
  2026-09-06 — see §13); stop when validation macro accuracy fails to improve
  over 2 consecutive evaluation points (patience unchanged). Every compared row
  gets **equal checkpoint-evaluation opportunities**: both `train_student.py`
  (CE) and `train_kd.py` (KD) validate on the full val split at the end of
  every epoch, through the same `evaluate.score_predictions` path every
  reported number in this project goes through — never on the training loss,
  which is not comparable across CE and KD by §4's own fairness rule. The
  checkpoint kept is the epoch with the best validation macro seen, not
  whichever epoch triggered the patience stop (`distillation/epoch_loop.py`,
  `EarlyStopper`).
- **Numerical failure** (loss NaN/inf, 0% task accuracy, gradient explosion) is an
  implementation defect to diagnose — **never** reported as a research finding
  about the method. Cf. the historical all-zero KD tables (`NEW_SUBMISSION.md` §4.1).
- **Drop an optional experiment** when: its compatibility gate (§8.1.1) fails; its
  CE baseline is not usable; or its measured cost exceeds the remaining budget.
  Dropping is recorded in §13 with the reason — silent omission is prohibited.
- **Scope reduction under pressure** is transparent: state the reduction in §13 and
  delete the claims it no longer supports. As of 2026-09-07 every row runs at a
  single seed (§5), so the reduction available here is over *rows*, not seeds.

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
| D3 | F → S2: CE only **(not run — §13, 2026-09-07)** | depth | ✅ (S2) | – | – | – | ✅ (F) |
| D4 | F → S2: CE + raw KD | depth | ✅ | – | ✅ | ✅ | ✅ |
| D5 | F → S2: CE + LoCa KD | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
| D6 | P → S2: CE + LoCa KD *(originally the intended recipe; superseded by D7, author decision 2026-09-09 — §13)* | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
| D7 | Joint feature + CE + LoCa KD *(selected as the primary/reported recipe, author decision 2026-09-09 — §13: beats D6 56.82% vs. 53.50% val)* | depth | ✅ | ✅ | ✅ | ✅ | ✅ |
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

Identical for every model, embedded in each model's **native** chat template.
**Frozen as the `terse` style** (amended 2026-09-06, §13):

```
Answer in one or two words. No explanation.
```

This **replaces** the legacy "single word or number" instruction, which conflicts
with legitimate multiword answers such as "tissue box". It also supersedes an
earlier draft of this section that named a longer, answer-space-enumerating
prompt (`"...yes, no, left, right, or the object name..."`) — that text was never
actually run; every measured number to date (B1–B5, teacher screens, Gemma QAT
screen, all of §13's amendments) used `terse`, and enumerating the answer space
in-prompt would leak task structure a real deployment does not have. `terse` is
what both `distillation/train_student.py` and `evaluation/zero_shot_inference.py`
render, and it is what the enable_thinking template fix (§13, 2026-09-06) was
validated against.

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
| Historical teacher | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | ~8 B | `LlavaOnevisionForConditionalGeneration` |
| Historical student | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | ~0.9 B | `LlavaOnevisionForConditionalGeneration` |
| Deferred | `Gemma 4 12B → MiniCPM-V-4.6` | — | Not started for this submission |

The historical pair is the exact teacher/student checkpoint IDs hardcoded in the
thesis-era training and eval scripts (`distillation/knowledge_distillation7b_*/*/train_online_kd.py:76-78`,
`inference/inference_utils.py:24-26`, `evaluation/onevisionv3/evaluate_onevision.py:41-43`
— all consistent; the `lmms-lab/llava-onevision-qwen2-7b-ov` ID appears only in one
retired, non-hardcoded datamodule variant and was not used by the actual training runs).
It gets zero-shot depth and RGB reference runs plus one matched depth-CE fine-tune,
in the same shape as B1/B2/B3, scored on frozen v2.4 with the repaired evaluator.
It is a historical-reproduction control, not a portfolio competitor: never
promoted to a 3-seed row, never placed in the main comparison table.

**Historical recipe IDs** (mirror B1/B2/B3/T1/T2 exactly, same val split, same
frozen v2.4 release, same repaired evaluator, `terse` prompt style since §9.5 of
`pilot_findings.md` already settled that wording for this evaluator/model-size
range):

| Recipe | Model | Modality | Training | Mirrors |
|---|---|---|---|---|
| `H-T1` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | depth | none, zero-shot | `T1` |
| `H-T2` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | rgb | none, zero-shot | `T2` |
| `H-B1` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | depth | none, zero-shot | `B1` |
| `H-B2` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | rgb | none, zero-shot | `B2` |
| `H-B3` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | depth | LoRA r16, lr 1e-5, 1 epoch, seed 17 | `B3` |

All five are PILOT rows under the same §9.5 Option A rule as everything else on
this machine: never promoted into a confirmatory or main table without a
fair-tuning-budget re-run under the §5 seed policy and the §7.3 epoch policy.

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
| G1 dataset | v2.4 hash verification passes; five-type contract frozen; construction claims trace to release artifacts | ✅ verification passed 2026-09-05; re-verified on the 4090 server 2026-09-05 (drift only in gitignored `data/index/`, resolved by re-running `build_index.py`); `dataset_protocol.md` outstanding |
| G2 evaluator | Edge cases verified; random baseline fixed; ID integrity locked | ✅ **fix confirmed in code and by re-run** — `evaluate.py`'s `parse_answer_space(...) or open_vocabulary` no longer collapses to `"nan"`; `python -m pytest tests/ -q` passes (172 passed, 3 skipped) and `evaluate.py --baselines-only --split val` reproduces sane per-type baselines on the 4090 server, 2026-09-05. Formal sign-off against every §6.2 edge case in `NEW_SUBMISSION.md` still pending an explicit review pass |
| G3 training | Tested losses, correct masks/alignment, valid contrastive candidates, depth-only inference | 🟡 **partially closed** — `tests/test_losses.py`, `test_xtoken.py`, `test_depth_input.py`, `test_inference_isolation.py` (synthetic-tensor unit tests) pass on the 4090 server, 2026-09-05. Still open: the 32-example overfit smoke test and cached-vs-live teacher agreement check against real checkpoints (blocked on model downloads completing) |
| G4 resources | Pilot + resume fit the actual GPU with measured throughput | ✅ **closed 2026-09-06** — teacher cache generation measured 20.61 GB peak / 7.53 ex/s on real batch-4 depth-substitution batches (§13.1 point 2); student CE training measured 12.84 ex/s at batch-4 without gradient checkpointing (`pilot_findings.md` §10); both fit the 4090 with headroom above the §9.4 ≈2 GB target |
| G5 central benefit | KD gain over strongest matched CE with paired uncertainty (single-seed since 2026-09-07, §5 — the three-seed condition this row originally stated is superseded) | ✅ **closed 2026-09-09/10.** Test split scored once, after settings locked (§5, §13 2026-09-09 rows). **D7 (primary/reported recipe) 54.1% test vs. B3 (primary comparator) 43.6% test — a +10.5-point gain, paired cluster bootstrap 95% CI [+9.6, +11.4], excludes zero in all 10,000 replicates** (`evaluation/paired_bootstrap.py`, seed 20260905 per §5; `runs/kd/confirmatory_recording/D7_vs_B3_test_bootstrap.json`). Clears §5.1's ≥2-point-and-CI-excludes-zero bar by a wide margin. Val-to-test gap was small and in the expected direction (D7 56.4%→54.1%, B3 44.9%→43.6%) |
| G6 component rationale | Each retained component beats its simpler alternative | 🟡 **partially closed, test evidence now in.** Alignment clearly matters: D4/D5/D6/D7 all beat B3 by 4.8-10.5 points on test with tight CIs excluding zero, while X2 (unaligned CE+KD) beats B3 by only **+1.0 point** (CI [+0.5,+1.5]) — statistically real but below the ≥2-point practical bar, i.e. **unaligned KD alone does not meet the study's own success threshold; the alignment stage is what does the work**. But the raw-vs-LoCa KD question is **not** resolved in LoCa's favor: D4 (raw KD, 50.6% test) vs. D5 (LoCa KD, 50.3% test) from the identical stage-F checkpoint — **CI on the paired difference is [-0.2, +0.8], includes zero** (`runs/kd/confirmatory_recording/D4_vs_D5_test_bootstrap.json`) — indistinguishable on held-out data. This **triggers §12's predeclared outcome rule** ("Raw KD matches LoCa → remove LoCa from the central method claim") for the D4/D5 pairing specifically. It does not by itself say anything about D7 (joint, not F→S2 — no "D7 with raw KD instead of LoCa" row exists) or D6 (P→S2, likewise untested against a raw-KD twin), so the manuscript's claim about D7 cannot cite D4-vs-D5 as LoCa's justification — this is a real, open gap, not resolved by what has been run. Still missing: D2 (CE+LoCa, unaligned — training 2026-09-09) and D8 (cosine feature vs. contrastive — training 2026-09-09) to complete the alignment-loss and KD-loss ablation matrix; D3 (CE-only on aligned vision, would isolate stage-F's own contribution) still not run on SUN-RGB-D by author decision (§13, 2026-09-07) |
| G7 generalization | Independent scenes + second-family evidence support the stated scope | ❌ open — LOSO (train on 3 of 4 SUN-RGB-D sensors, test on the held-out `realsense`) shows no measurable drop (`pilot_findings.md` §14), but that is cross-*sensor*, not cross-*dataset*. The one cross-dataset attempt, ARKitScenes (Phase 5), is **not** included in this submission by author decision 2026-09-08 (`arkitscenes_plan.md` §8 point 5): single-seed, ~33-38-item val/test, D5 came back *below* D3 (30.95% vs 40.95%, contradicting the core hypothesis) and Phase 4's human audit of the release never ran. No second-family student track (e.g. SmolVLM2) has been run either (`arkitscenes_plan.md` §7: "R2.1 ... not addressed") |
| G8 submission readiness | Every major claim has a verified artifact | 🟡 partially closed — (1) ~~test split unscored~~ **done 2026-09-09/10 for the primary set** (B3/B5/X2/D4/D5/D6/D7/D9 — see G5/G6 above); D2/D8 training 2026-09-09/10, to be test-scored the same way once trained (§13 rows this date); (2) ~~primary-claim recipe undecided~~ **decided: D7**, confirmed on test (G5); (3) D2, D8 training in progress; ~~B6~~ **dropped** (§13.1 pt 4); (4) ~~paired-uncertainty instrument~~ **implemented and run** (`evaluation/paired_bootstrap.py`) — see G5; (5) ~~stage-F reproducibility gap~~ **fixed** — see §13 row this date. Remaining before this gate can fully close: D2/D8 test scores, and a manuscript-level reconciliation of the D4-vs-D5 raw/LoCa finding (G6) |

### 11.1 Known blocking defect (Phase 3) — RESOLVED 2026-09-05

~~`evaluate.py:189` computes `str(row["answer_space"] or "")`. For open-vocabulary
types `answer_space` is `NaN`, and **`NaN` is truthy in Python**, so the `or ""`
fallback never fires. `str(nan)` → `"nan"` → `options == ["nan"]`, and the
`open_vocabulary` fallback on line 191 is unreachable. Every open-vocabulary row
"predicts" the literal string `nan`.~~

~~Observed consequence in the current baseline output: `identify_superlative`
random = **0.0%**, `nearest_object` random = **0.0%**. Correct NaN handling
already exists nine lines earlier in `snap_to_answer_space` (line 105).~~

~~**Fixing this will move the published 30.2% random macro figure.** No comparative
result may be reported until G2 closes.~~

**Fixed and verified 2026-09-05** (§11 G2 row): `parse_answer_space(...) or
open_vocabulary` no longer collapses to `"nan"`; `identify_superlative` and
`nearest_object` no longer read 0.0% random. This subsection is kept for the
historical record of the defect, not as a live blocker — see §11 G2 (✅) for
current status. Every comparative number reported after 2026-09-05 already
reflects the fix.

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
| 2026-09-05 | §9.2 | Added the thesis/rejected-submission OneVision pair (`llava-hf/llava-onevision-qwen2-7b-ov-hf` teacher, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` student) to the portfolio as a **historical-reproduction row**: zero-shot depth/RGB references plus one matched depth-CE fine-tune, scored on frozen v2.4 with the repaired evaluator, in the same shape as B1/B2/B3. Never a headline result and never promoted to 3 seeds — its purpose is to separate what changed because of the new dataset/evaluator from what changed because of the new model portfolio | Author request, to see how the exact old model/recipe performs once the dataset and scoring defects are fixed | No |
| 2026-09-05 | §9.1, §11, §14 | **4090 is back, ahead of the 2026-09-06/07 estimate.** Fresh server: environment rebuilt from scratch (`.venv-models`: torch 2.14.0+cu126, transformers 5.16.1, accelerate/peft/bitsandbytes/etc.), `python -m pytest tests/ dataset/dataset_creation/v2/tests -q` re-run clean (172 passed, 3 skipped, 0 failed) confirming **G2 holds on this machine** — `evaluate.py --baselines-only --split val` reproduces sane open-vocabulary baselines (macro chance 30.3% / random 29.9% / majority 33.5% / question-only 33.7%; no more 0.0% on `identify_superlative`/`nearest_object`), matching `pilot_findings.md` §8's account of the fix. G1 re-verified: `freeze_release.py --verify v2.4` reported drift only in `data/index/` (expected — gitignored P0 output per `SERVER_SETUP.md`), resolved by re-running `build_index.py`; release CSVs and row counts untouched. `export_v1_schema.py` regenerated the v1-schema CSVs with the documented row counts. Full Qwen3.5-9B/0.8B, InternVL3.5-1B/8B, SmolVLM2-500M, Florence-2-large, and Gemma-4-12B-it portfolio download in progress (all seven checkpoints confirmed public/ungated on HF, no token needed). SUN RGB-D raw imagery archive was found truncated on this server (partial download, no EOCD) and is being re-fetched. | Author moved to the 4090 server; environment, dataset, and evaluator needed to be re-established from a bare machine | No |
| 2026-09-06 | G1, `freeze_release.py` | **Fixed the recurring `data/index/` false alarm rather than re-running around it a second time.** A second, unplanned server restart rebuilt `scene_index.jsonl` under a different numpy/scipy stack and `--verify v2.4` flagged drift again — the same symptom the 2026-09-05 row called "expected" and "resolved," which this shows was not durable. Investigated instead of assumed: the rebuild's P0 drop decisions matched the frozen release's `stats/drops.csv` `p0_drops` rows exactly (11,235/11,235, 0 differences), and `index/manifest.json`'s canonical hash (config, seed 42, toolbox checksums, per-type counts) was unaffected — so the content this file's freeze exists to pin was unchanged; only float-precision noise in per-object geometry/depth fields (`area_frac`, `centroid_x/y`, `depth_median_m`, `depth_valid_frac`) was not, because those are computed from numpy/scipy arithmetic whose last bits shift with the numeric stack. This is `_sha256_of_manifest`'s `built_at_utc` problem one level down, so it gets the same fix: `freeze_release.py`'s `sha256_of()` now canonicalises `scene_index.jsonl` (round floats to 6 dp — several orders below every P0/P2 threshold this pipeline acts on, e.g. `min_area_frac=0.005`, `min_valid_fraction=0.3` — sort keys, preserve record order, since `build_index.py` iterates the toolbox `.mat` deterministically with no `set()`/`glob`/multiprocessing and a reordering would be real content) before hashing, rather than hashing raw bytes. `FROZEN_v2.4.json`'s recorded hash for this file is updated to the canonical value under a new `corrections` entry inside that manifest (dated, with the old hash and the reasoning above) rather than silently overwritten — "immutable once frozen" (§10) means the release content, not a verification-tool bug discovered after the fact; this is the same class of correction as the G2 evaluator fix, applied to a checksum instead of a metric. `--verify v2.4` now passes clean, and a negative control (mutating one record's `raw_name`) confirmed genuine content drift is still caught. 6 new tests pin the behaviour (`tests/test_freeze_release.py`); full suite 175 passed. | Author asked what the best fix was after the false alarm recurred; re-running `build_index.py` again would only postpone the next recurrence | Yes — verify clean before and after, negative control still fails, full suite green |
| 2026-09-06 | §7.3 | **Implemented the epoch/early-stop policy §7.3 always declared but neither training script ever ran.** Every CE and KD run to this point (B3, B5, X2) trained a fixed single epoch — the scripts' own CLI default, never checked against validation performance, and not what §7.3 specifies (up to 5 epochs, stop after 2 non-improving evaluation points). Author raised the cap to **10 epochs** for this run, patience unchanged at 2. `distillation/epoch_loop.py` (new, 8 tests) is shared by both scripts so "equal checkpoint-evaluation opportunities" is enforced by construction rather than by convention: both validate on the **full** val split at the end of every epoch through the exact `evaluate.score_predictions` path every reported number in this project already goes through — not the training loss, which §4 already forbids using to compare CE and KD. The kept checkpoint is the best-scoring epoch, not whichever epoch triggered the stop; `EarlyStopper`'s tests specifically cover the case where accuracy peaks then declines for `patience` epochs, since that is the case a naive implementation gets wrong silently. Full suite 183 passed before the first multi-epoch run was launched. | Author asked whether the fixed-epoch runs so far were even necessary to run at full 3-seed rigor before checking whether more training changes the picture; single-epoch training had never been validated against §7.3's own stopping rule | No — declared and implemented before any multi-epoch result was seen |

| 2026-09-07 | §5, §7.3 | **Confirmatory seed set reduced from three (17, 42, 2026) to a single seed (17)** for every row, at author instruction ("we are over engineering with the seed thing, we should just pick one seed and continue with all our experiments with that seed"). What this buys: roughly 3x the rows per unit of GPU time — a matched B3/X2 pair costs ~4.6 h, so three seeds of the ladder was never affordable alongside the unbuilt LoCa and feature-alignment stages. What it costs, stated plainly: **retraining variability is no longer estimated at all.** The paired cluster bootstrap over scene groups still yields an interval, but it is an evaluation-set interval; §5's own row already warned that "a test bootstrap alone does not estimate retraining variability," and that warning now has no counterpart measurement to sit beside. §5's Joint-estimate row (hierarchical bootstrap over seeds x scene groups) is withdrawn as unsatisfiable. The manuscript must carry this as an explicit limitation. One empirical anchor survives and should be reported rather than buried: B3 depth CE was run twice under the multi-epoch policy before this decision, seed 17 = 44.9% and seed 42 = 44.0% val macro, a **0.9-point spread from seed alone** — larger than the 0.5-point CE-vs-KD gap measured at seed 17. That single observation is why sub-point differences in this study are not interpretable at one seed, and it does not go away by having stopped measuring it. It does not block the primary claim, because §5.1's threshold is a **>= 2-point** gain: a 0.5-point gap fails that threshold at any seed count, so seed replication could not have rescued it. The seed-42 row (`20260907-none2qw08b-B3-s42-594b204b`) stays recorded in `runs/INDEX.md` — §7.3 prohibits silent omission — and is cited as this spread estimate, not as a second seed of a two-seed design. | Author instruction, on the grounds that three-seed replication was over-engineering for this study's stage | **Yes — and this is an amendment made after seeing results, so per this section's own rule, exactly what was seen: validation macro for B3 s17 44.9%, B3 s42 44.0%, X2 s17 44.4%, B5 s17 62.2%, all under the 10-epoch/patience-2 policy. No test-split result has been observed; the test split remains untouched** |

| 2026-09-07 | §9.3, §8.2, §9.2 | **Feature-alignment path built, and the declared feature layer fixed: `visual.pooler_output` (the post-merger vision sequence, in language-model space).** Three candidates existed; the choice is recorded because §12 makes `feature_layer` part of a row's identity. Pre-merger `last_hidden_state` (student 768 / teacher 1152) is the closest analogue of the legacy `vision_tower.vision_model.post_layernorm` hook, but Qwen3.5 has no `post_layernorm`; a language hidden state was rejected because it mixes question text into a supposedly visual feature and would make the cache prompt-dependent. `pooler_output` was chosen as the representation the language model actually consumes, and because its width equals each model's text hidden size. Verified equal to a forward hook on `visual.merger` on the real 0.8B. Pooling is a float32 mean over each image's own merged tokens, then L2 normalisation; the packed `[sum_i tokens_i, D]` layout is split with `image_grid_thw` and the split is checked against the tensor, because a wrong split silently averages one image's patches into another's feature. Widths differ (1024 vs 4096) and both feature losses require equal widths, so a **bias-free linear alignment head** maps student -> teacher; it is applied to the student side only (a learnable map on the target could lower the loss by degrading the target) and is not part of §6's deployable student. Teacher features are cached **per image, not per row** — 4,187 distinct images behind 15,278 train rows — and the cache is **prompt-independent** (no chat template is rendered, so `prompt_hash` is deliberately null and the §10.4 `<think>` defect could not have affected it). Measured: 4,187 images in **1.4 min at 49.7 img/s, 31.85 MB, 19.44 GB peak**, digest `683d9c59eac4a6d9`. The contrastive negative bank implements the audit's B4 requirement — 255 negatives drawn from 255 **distinct scenes**, excluding every sequence present in the batch (109 of this split's 3,231 sequences hold more than one image, one holds 52, so a different image of the same room would be a false negative), with the distinct-scene count logged per step and an all-neighbour candidate set made a hard error. **D0's trainable surface corrected from `language_attention` to `vision_attention`.** As declared, D0 trained **zero student parameters**: a pooled vision feature loss has no gradient path to language attention, measured 0 of 48 LoRA tensors receiving gradient with only the alignment head updating — it would have logged a falling loss and saved a checkpoint behaviourally identical to the base model, the same class of silent no-op as the audit's B4 finding. §8.2's stage-F definition is what the corrected surface declares. LoRA surfaces are now anchored **regexes over fully-qualified module names** rather than suffix lists, because PEFT matches list entries with `endswith` and the vision blocks' attention output is named `proj`, which `q_proj`/`k_proj`/`v_proj`/`o_proj` all end with — a list would have silently wrapped language attention while the row reported itself vision-only. Verified the new language regex selects the **identical 24 modules** the old suffix list did, so X2's completed run stays reproducible, and the vision regex selects 24 vision modules with zero language leakage. Stage-F smoke test on real data: loss starts at the ln(256) = 5.545 chance floor and falls monotonically to **4.98** over 175 steps at 56 ex/s and **3.91 GB** peak — far cheaper than a CE/KD row's 14.5 GB, because a pure stage-F row never runs the language model. 29 new tests (`tests/test_features.py`), full suite 224 passed. | Author asked for the feature-alignment path to be built, it being the remaining honest test of the KD premise after X2 failed to beat CE | Partly — the §10.2 val results (B3 44.9%, X2 44.4%, B5 62.2%) were already known and motivated this work, but **no feature-row result exists**: the only numbers seen from this path are the smoke-test loss trajectory above, on 3,000 train rows with an 8-row val limit, which selects nothing |

| 2026-09-07 | §8.2, §9.2, §7.3 | **Two-stage rows (D3/D5/D8) implemented as F -> S2, resolving §13.1 point 5 in favour of option (a).** The reading was already in the protocol: §7.1 describes D3–D9 as two-stage rows where "stage one is a separate training pass before S2 begins", and §8.2 defines S2 as vision-frozen — so a row's feature objective belongs to its F stage, which is also the author's own two-phase method. The alternative (joint single-pass with a vision surface in S2) was rejected because it contradicts §8.2's S2 definition. **The stages are derived from the declared row, not added as new library rows**: `RecipeConfig.stage_one()` returns the row's feature objective on a `vision_attention` surface with no CE/KD, and `stage_two(parent)` returns its answer objectives with `feature_objective` cleared and `parent_checkpoint` set. So `D8` remains one id meaning "CE + X-Token + LoCa + cosine alignment" in the §9.2 matrix, D5 and D8 stay distinguishable by their stage-one objective (contrastive vs cosine), and — deliberately — **no field was added to `RecipeConfig`**, because `record_run.build_configuration` folds `resolved()` into the run-id hash and a new field would have silently orphaned X2's completed run id. Stage one's LoRA is **merged into the base weights** before stage two attaches its own (`merge_and_unload`), so §8.2's vision freeze is structural rather than a flag that can be forgotten, and PEFT never holds two adapters with only one trainable. **CORRECTED same day, on author challenge — the original text of this row claimed a stage-F row "cannot answer a question and has no validation macro," which is wrong.** It scored 0.06% only because `--val-limit 8` was set and `score_val_macro` scores whatever it is handed against the **whole** 1,720-row gold split, so 8 predictions read as 8/1720; the same artifact had already produced a spurious 7.1% in an earlier `--limit 300` run. Measured at full coverage, the stage-F checkpoint answers all 1,720 rows and scores **34.3%**, directly comparable to B1's **36.18%** zero-shot depth (same language model, unaligned vision), against a 30.3% chance floor and a 33.7% question-only baseline. The vision-alignment stage therefore **does** score the full val split every epoch, and the curve is recorded in `val_history`, which is what allows the alignment budget to be chosen from evidence rather than guessed. It still does not *select* the checkpoint, for a reason that survives the correction: the number measures how legible the shifted vision is to a language model that has **not** adapted to it, which can diverge from how good a starting point that shift is once S2 adapts — and the first measurement is consistent with that caution, coming in 1.9 points *below* unaligned zero-shot after a deliberately under-trained 400-row pass. So stage F runs a declared budget (`--alignment-epochs`, default 3), keeps its **last** epoch, and additionally reports `best_reading_epoch` as evidence; `resource_usage.json` records `patience: null` and `stopped_on: "fixed_budget"`. The recurring coverage artifact is now guarded at source: `score_val_macro` announces partial coverage loudly and says the number must not be quoted (2 new tests). Naming also corrected on the same challenge: "F" is the §8.2 stage id and stays as the hashed identifier, but every human-facing name is now the **vision-alignment** stage (`--alignment-epochs`, `--alignment-learning-rate`, `--skip-alignment`, with the old `--f-*` spellings kept as aliases). A single-pass invocation of a two-stage row is refused with the two-stage instruction. Verified end to end on real data (D8, 400 rows): stage F cosine loss 0.985 -> 0.877 at 53 ex/s and 3.90 GB, then stage S2 merged that adapter, froze vision, and trained CE 1.71 -> 1.40 with X-Token KD 3.01 -> 1.96 at 11.5 ex/s and 13.74 GB. **This also closes the deferred LoCa GPU smoke test** — the LoCa path ran for the first time on a real GPU and reported `loca_gold_found_rate 1.0000`, i.e. gold was present in the cached top-K at every supervised position, so no row was dropped from calibration. 7 new tests, full suite 235 passed. | Author sign-off on the recommended option, to build the F->S2 chaining | Yes, in the same sense as the rows above: §10.2's val results were known, but **no two-stage row result exists** — the only numbers seen from this path are the 400-row smoke-test loss trajectories above, which select nothing |

| 2026-09-07 | §8.2, §9.2 | **Vision alignment (stage F) also trains the merger, not just vision-tower attention** — author decision. The merger (`visual.merger.linear_fc1`/`linear_fc2`) is the model's own vision->language projector: leaving it fixed caps how much the alignment loss can reshape the vision->language interface itself, versus only reshaping what happens upstream of it. Verified before adding it: every one of the 12 vision blocks' MLPs is *also* leaf-named `linear_fc1`/`linear_fc2`, identical to the merger's, so the new `vision_merger` LoRA target is anchored on `visual.merger` specifically rather than the leaf name — a suffix-style match would have silently trained all 12 block MLPs alongside the one merger asked for, the same collision class `o_proj`/`proj` already was for vision vs language attention. Checked on the real model: the regex selects exactly the 2 merger modules, none of the 12 block MLPs, and composes with `vision_attention` to select all 26 as expected. `stage_one()` and D0's declared surface both updated to `("vision_attention", "vision_merger")`. This changes nothing about how stage F is read: it is still not required to beat CE on its own (§13, 2026-09-07 amendment above), and the frozen-LM val number it reports is still diagnostic, not a selection criterion — only the trainable surface changed, which affects what a downstream S2 stage inherits. A merger-enabled contrastive alignment run is queued (`align_curve_contrastive_merger_s17`) for comparison against the vision-attention-only curve already recorded. 1 new test, full suite 238 passed. | Author decision, on the reasoning that the merger is a learnable projector rather than part of the frozen pretrained decoder and should not be exempt from alignment | No — queued but not yet run at time of writing |
| 2026-09-07 | §8.2, §9.2 | **Merger-trainable vision alignment measured and rejected.** The queued run from the row above (`align_curve_contrastive_merger_s17`, `("vision_attention","vision_merger")` surface) completed: full-val reading curve **25.84 / 24.27 / 23.68 / 22.86 / 22.92%**, monotonically declining after epoch 0 and **below the 30.3% chance floor at every epoch** — worse than doing nothing. This is the opposite outcome from the attention-only curve trained the same way (34.48–35.40%, §12 of `pilot_findings.md`), which stays well above chance throughout. Reading, not selecting (§13's coverage-fix amendment above still applies), so this does not itself rule the surface out for S2 — but it is consistent with a plausible mechanism: the merger is the fixed interface the frozen language model has already learned to read, and perturbing it moves the language model further from a representation it was never asked to adapt to, rather than closer to one it was. **No S2 stage was chained onto this checkpoint.** D5 (§12 of `pilot_findings.md`) uses the attention-only checkpoint (`align_curve_contrastive_s17`), not this one. | Measured; no author decision needed beyond noting the result | Yes — the curve above is exactly what was seen |
| 2026-09-07 | §8, §10.2 | **D5 run and scored: 50.13% val macro, the first row in this study to beat CE and to clear §5.1's ≥2-point bar** (B3 44.9%, X2 44.4%, D5 50.13% — +5.2 / +5.7 points, seed 17, matched 10-epoch/patience-2 policy, matched LoRA rank). Stage F used the attention-only alignment checkpoint from the 2026-09-07 row above (`align_curve_contrastive_s17`, `best_reading_epoch=1`, 35.40% full-val reading), not the merger-trainable one, which the same-day row above rejected. Full curve and setup recorded in `pilot_findings.md` §12. **Author decision, same day: D3 (row 8's CE-only-on-aligned-vision case) will not be run.** D3 exists in §8's table to attribute D5's gain between stage F (vision alignment) and stage S2 (answer-distribution KD); the author's instruction is that the manuscript should present the two-stage pipeline as a single knowledge-distillation method rather than decompose it — stage F transfers the teacher's visual-feature structure into the student, which is distillation under this protocol's own §1 framing, so it is not treated as a separate, non-KD ingredient that needs isolating. Consequence stated plainly: the manuscript can report that the *combined* two-stage method beats CE by 5.2 points, but cannot say how much of that is attributable to alignment versus to KD proper. §8's D3 row is left in the table as a label-access definition (for anyone who later chooses to run it) but is marked not-run in the note column. | Author instruction: "I want my paper to believe that that 5.2 is from KD and also the alignment part is also a kind of KD so it's no biggie" | Yes — the D5 result above was known when the decision not to run D3 was made |
| 2026-09-07 | §8, §8.2, §13.1 pt 5 | **D4, D6, D7, D9 added to `recipe_library()`** — declared but not run. **D4** ("F -> S2: CE + raw KD") is D5 without LoCa: same contrastive stage-F alignment, `use_loca=False` in S2. **D7** ("Joint feature + CE + LoCa KD") is deliberately *not* routed through the F->S2 machinery: it declares `stage="joint"` with both `vision_attention` and `language_attention` trainable at once, so `is_two_stage()` (previously `feature_objective != "none"`) was changed to also require `stage != "joint"` — otherwise D7 would have been incorrectly forced through stage-F/stage-S2 chaining it never asked for. This is the §13.1 point 5 option-(b) shape, deliberately kept available as its own row rather than reopening that decision for D3/D5/D6/D8. **D6** ("P -> S2: CE + LoCa KD") needed real plumbing, not just a config entry: §8.2 defines **P** as stage one "as submitted" — feature alignment *plus* a small raw KD term, distinct from D5's clean feature-only F stage. A new `RecipeConfig.stage_one_p()` keeps `kd_objective` in place (`stage_one()` clears it) and forces `use_loca=False` (§8.2 calls P's KD term specifically *raw*, and D6's own S2 stage using LoCa does not change that). No new plumbing was needed in `compose_loss` — it already calls `adapter.student_logits` whenever `kd_objective` is set regardless of `stage`, so the KD term's gradient already flows back through the frozen language decoder into the trainable vision surface (frozen means no optimizer step for those parameters, not no backward pass through them) — the same mechanism that makes stage F's feature term reach vision parameters, extended to a second loss term. The "small" weight is not fixed by the protocol, so `--p-lambda-kd` (default 0.1) is an explicit, overridable author-facing choice rather than a measured one. `train_kd.py --stage` gained a `P` choice and `train_two_stage.py` now runs D6's stage one as P (into a `stage_P` directory) rather than F, passing it `--cache` and `--p-lambda-kd` since P — unlike F — needs the top-K logit cache. **D9** ("F -> S2: teacher-prefix raw KD only") is added at the config level only (`use_ce=False`, `use_loca=False`, `kd_objective="xtoken"`, `feature_objective="contrastive"`) — §8.1's strict label-access rule needs the gold answer column removed before the training/cache interface and replaced by a teacher-generated prefix, which needs a teacher-completion cache that does not exist yet (a new artifact, distinct from `build_teacher_cache.py`'s top-K logit cache) and a batch-builder path that consumes it. The config's `notes` field says `BLOCKED` so this cannot be mistaken for run-ready. 5 new tests (`tests/test_pipeline.py`): the matrix-coverage test now checks D3/D4/D6/D7/D9 too, plus dedicated tests for D7's `is_two_stage()==False`, D6's `stage_one_p()` (kd_objective preserved, `use_loca=False`, default and overridden `lambda_kd`), the refusal for a feature-only row with no `kd_objective`, and D9's declared shape. Full suite 277 passed (243 + 34 dataset tests). | Author instruction: "start on it" (add D4/D6/D7/D9 to `recipe_library()`), part of the GPU-task queue authorized the same session | No — none of the four rows has been run; this is config and orchestration only |
| 2026-09-07 | §9.2, §9.5 | **Gemma-4-12B-it screened at both precisions the portfolio holds: unusable as a teacher on this benchmark.** Zero-shot, full val split, both modalities: plain (non-QAT, NF4) scores **28.2% depth / 37.8% RGB**; the earlier QAT-dequantized screen scores 32.8% depth / 39.4% RGB. Both are below Qwen3.5-0.8B's own untrained zero-shot (B1 36.18% depth, B2 42.61% RGB), and the plain checkpoint's depth score is below the 30.3% chance floor, driven by a high invalid-answer rate (`left_right` 65.6%, `nearest_object` 44.9%). Full table and per-type breakdown in `pilot_findings.md` §13. Screening the plain checkpoint needed a real infrastructure fix, not just a run: `HF_HOME` does not persist across separate shell invocations in this environment, so the first three attempts silently redownloaded the model into the wrong default cache (`~/.cache/huggingface/hub`) and either hung for minutes with zero GPU memory allocated (online) or failed with a misleading `does not appear to have a file named ... model.safetensors` error (`local_files_only=True`/`HF_HUB_OFFLINE=1`) despite the file being fully present under the real cache. Root-caused via `wchan`/`futex_wait_queue`, confirmed by finding a 6.5 GB partial re-download under the wrong cache path; fixed by exporting `HF_HOME` inline in the same command as every subsequent GPU invocation. **Conclusion for §9.2/9.3: `gemma-4-12B-it → Qwen3.5-0.8B` stays a feasibility/mapping-quality screen only, never a headline KD pair** — confirmed independently at both precisions now, closing that question rather than leaving it resting on the QAT screen alone. | Author request: "what was the Gemma-4-12B's results? I hope its better than the Qwen3.5-0.8B", from an earlier session turn | Yes — both scores were measured before this row was written |
| 2026-09-07 | §9.3, §11 | **Gemma-4-12B-it → Qwen3.5-0.8B X-Token mapping built and verified — this is the real stress-test pair §9.3 named.** CPU-only, no GPU needed: `build_vocabulary_mapping` gives **coverage 1.0** (all 248,077 student tokens map to something) but **exact-match fraction only 56.66%** (vs 99.62% for Qwen↔Qwen, §11) — confirming this pair genuinely tests generalisation beyond related tokenizer families rather than being a formality. Most-collapsed teacher id absorbs 5,880 distinct student ids (vs 665 for Qwen↔Qwen). Re-ran §11's `scatter_add_`-accumulates check against it: the raw result (0.999972 vs 1.0) read as a failure against the existing `1e-5` tolerance, but is float32 summation noise from the much larger collapse, confirmed three ways (a float64 rerun of the identical call lands at 0.99999999999993; a plain sequential float32 sum of 5,880 copies of 1/5,880 reproduces the exact same erroneous value bit-for-bit; both converge to 1.0 in float64/arbitrary precision). Fixed `check_scatter_add_accumulates`'s tolerance to scale with collapse size (`max(1e-5, n_collapsed * 1e-8)`) rather than leaving a constant tuned on one pair's smaller collapse. 2 new tests (`tests/test_xtoken.py`: the scaled tolerance accepts a synthetic 6,000-way collapse; a genuinely broken projection — half the mass reaching the target — still fails regardless of collapse size). Full suite 279 passed. Projection correctness is now confirmed on this pair too; what remains open is empirical, not a code question: whether a 57%-exact-match mapping carries enough signal for X-Token KD to help once this pair is actually trained — separate from, and not resolved by, §13's finding that Gemma-4-12B is a weak teacher here regardless of mapping quality. | Author instruction: do item 14 (X-Token mapping check) after items 8-11, done here since it needed no GPU and did not contend with the GPU queue | Yes — the mapping statistics were measured before this row was written |
| 2026-09-07 | §7.3, §8, new | **Leave-one-source-out (LOSO) generalization probe: training on 3 of 4 SUN RGB-D sensors transfers to the 4th (`realsense`) with no measured drop.** B3 (depth CE, seed 17) retrained on a split excluding every `realsense` row from training (13,793 rows), stopped by patience-2 at epoch 3 of 6 run. In-distribution val (3 sensors, 1,565 rows): 43.57%. Held-out `realsense` val (155 rows, 0 seen in training): **46.4%** — nominally higher, but at n=155 the honest claim is "no measurable degradation," not "generalizes better." Full breakdown in `pilot_findings.md` §14. Needed a real fix, caught mid-run rather than after: `score_val_macro` (`distillation/epoch_loop.py`) had no way to know a caller's val CSV was a genuine, deliberately-smaller gold set rather than incomplete coverage of the frozen release — the first run scored the 1,565-row in-distribution predictions against the full 1,720-row frozen `val.csv` and silently counted the 155 excluded rows as wrong every epoch (37.24% epoch-0, discarded, never recorded as a result). Added a `release_dir` parameter to `score_val_macro` (defaults to the frozen `RELEASE_DIR`, so every existing caller is unaffected) and a `--val-release-dir` flag to `train_student.py`, which now refuses to launch with a subset `--val-csv` unless `--val-release-dir` is also given. 2 new tests (`tests/test_epoch_loop.py`): the override scores against the smaller gold set without flagging PARTIAL COVERAGE; the default stays the frozen release. Full suite 281 passed. Sensor split created once as `distillation/loso/{held_realsense,indist_no_realsense}/{train,val}.csv`, derived from the frozen release's own `sensor` column, not a new annotation. **Same caveats as every other single-seed row in this study (§7.3): one seed, one excluded sensor, not a full cross-sensor matrix.** | Author instruction: "start on it" (item 9, leave-one-source-out study), part of the GPU-task queue authorized the same session | Yes — both split scores were measured before this row was written |
| 2026-09-08 | §8, §13.1 pt 5 | **D4 run: raw KD beats LoCa KD from the identical aligned checkpoint.** D4 reused D5's own stage-F adapter directly (`--parent-checkpoint runs/kd/align_curve_contrastive_s17/preserved/epoch_1`, same run, not re-derived) so the two rows' vision-alignment weights are bit-identical; D4's stage S2 differs from D5's only by `use_loca=False` (raw KD vs. LoCa KD). Best val macro **51.59%** (epoch 5 of 8 run, patience-stopped, 202.1 min, 10.08 ex/s) vs. D5's recorded 50.13% (epoch 7 of 10) — a **1.46-point gap, one seed each**, from a fixed alignment stage. This is not the alignment-vs-KD attribution question the manuscript declined to answer (§13, 2026-09-07 author decision) — both rows include the same alignment stage; this isolates the S2 calibration choice alone. Full writeup `pilot_findings.md` §15. | Author instruction: "start on it" (item 11, run D4/D6/D7/D9 once defined), part of the GPU-task queue authorized 2026-09-07 | Yes — both val macros were measured before this row was written |
| 2026-09-08 | §8, §8.2 | **D6 run — the method "as submitted" — beats both D4 and D5.** D6 is the P → S2 row: stage one is feature alignment *plus* a small raw KD term (`--p-lambda-kd 0.1`, §8.2), not the feature-only F stage D4/D5 share; stage S2 (CE + LoCa KD) is otherwise the same shape as D5's. Best val macro **53.50%** (epoch 3 of 6 run, patience-stopped) vs. D4's 51.59% and D5's 50.13%. Interrupted once mid-run by a planned server restart (§ new row below documents the restart-safety infrastructure) — stage P had already finished and checkpointed cleanly (92.4 min, 3 epochs, 8.26 ex/s) before the restart, so only stage S2 needed to rerun from epoch 0 after resuming; the resumed S2 run matched the pre-restart attempt's epoch-0 val macro closely (50.70% vs. 50.82%), consistent with the same seed and data. Total pipeline compute: 92.4 (P) + 150.6 (S2, resumed run) = 243.0 min. Full writeup and cross-row table `pilot_findings.md` §16. D9 remains the only recipe not run (still `BLOCKED`, §8.1) — D4, D5, D6, D7 are now the complete run set for this GPU-task queue. | Author instruction: "start on it" (item 11, run D4/D6/D7/D9 once defined), part of the GPU-task queue authorized 2026-09-07 | Yes — all val macros were measured before this row was written |
| 2026-09-08 | infra | **Restart-safety pattern for long training runs on a wall-clock-limited server.** Author's server enforces a ~12h wall clock, reset only by a manual restart; D6's P → S2 pipeline (243 min total) would not fit in the remaining window uninterrupted. Rather than let a restart kill an in-progress run destructively, stage P was allowed to finish and checkpoint naturally (`train_two_stage.py` already writes a finalized `<out>/stage_P/adapter` only at the end of that stage, via the same per-epoch-save-then-keep-best-then-finalize mechanism `train_student.py` uses for LOSO), then a self-contained resume script (`runs/kd/resume_d6_d7.sh`, written to persistent disk rather than the session-scoped scratchpad, since the session itself does not survive a restart) was prepared ahead of time: it checks `stage_P/adapter` exists before doing anything, then reruns S2 with `--skip-alignment` (skipping stage P entirely) followed by D7. No mid-epoch resume exists for S2 itself — any S2 progress made in a run that gets killed before it finishes is fully discarded and S2 restarts from epoch 0, so the practical rule this establishes is: **restart as soon as possible after a natural stage boundary, not as late as the wall clock allows** — every additional minute spent mid-S2 before a restart is compute that gets thrown away. This pattern (let a stage finish, snapshot a resume script to disk, restart, resume with `--skip-alignment`) generalizes to any future two-stage run under the same wall-clock constraint. | Author's own infrastructure constraint, surfaced mid-session ("my server usually has a wall clock for 12 hours...") | N/A — this is an infrastructure/process decision, not an empirical result |
| 2026-09-08 | §8.1 | **D9's code path is now built and ready to run — nothing left to unblock it but the cache-building GPU passes themselves.** Built while D7 trained (CPU-only work, no GPU contention): (1) `distillation/build_teacher_generation_cache.py` — new script, runs the teacher in free generation mode with no gold answer anywhere in the prompt (same `terse` prompt/decoding contract as every other measured number, §8.3), writes a consolidated `generated_text` cache with incremental save-every-N-rows so a wall-clock restart (see the row above) loses at most that many rows rather than the whole pass. (2) `build_teacher_cache.py` gained `--prefix-source {gold,teacher_generated}` — `teacher_generated` forces each row's cached completion instead of `row["answer"]` when building the topk_logits cache. Backward compatibility was the load-bearing constraint here: `CacheKey.digest()` reads every field with `.get(name)`, so the default ("gold") case encodes `prefix_source` as `None`, byte-identical to every digest computed before this flag existed — an explicit `"gold"` string would have silently invalidated the existing gold-prefix cache D4/D5/D6/D7 already depend on. Pinned directly (`tests/test_teacher_cache_loader.py::test_prefix_source_gold_default_produces_the_pre_existing_digest`). (3) `train_student.build_batch` was refactored into a thin gold-answer wrapper around a new `build_batch_with_answers(processor, rows, images, answers)`, which takes the forced text as an explicit `{question_id: text}` mapping rather than always reading `row["answer"]` — the single shared implementation both the gold and teacher-generated paths now go through, so the prompt-rendering/masking logic cannot drift between them the way the teacher-forced/eval prompt already once did (§8.3). (4) A new `GeneratedTextCache` reader (`teacher_cache_loader.py`, mirrors `FeatureCache`'s one-consolidated-table pattern) and `train_kd.py` wiring: `--generated-text-cache`, a `strict_label_access = (args.recipe == "D9")` branch that sets `prefix_source="teacher_generated"` in the topk_logits cache key, loads the matching generated-text cache, and — per §8.1's "must succeed with them absent," not merely unused — strips the `answer` column from every row before D9's training loop touches them. D9 still reuses D5's stage-F checkpoint like D4 does (feature alignment never touches answer text, so it needs no variant of its own). 12 new tests across `tests/test_train_student_batch.py` (new file) and `tests/test_teacher_cache_loader.py`; full suite 288 passed. **Still not run**: the actual free-generation pass (~15,278 rows through the 9B teacher) and the logits pass over its output — both need the GPU, which was occupied by D7 for the whole of this work. | Author instruction: "do it while D7 runs on gpu" (build the D9-unblocking infrastructure), following the author's own question about what D9 needed | N/A — no cache was built and no row was trained; this is infrastructure only |

| 2026-09-09 | §10, §11 | **D4, D5, D6, D7, D9 moved from ad-hoc `runs/kd/*/resource_usage.json` bookkeeping into proper `evaluation/record_run.py`-tracked, confirmatory run directories.** All five had `"pilot": true` in their recorded `resource_usage.json`/`recipe_config.json` because `--confirmatory` was never passed on the original training command line — a CLI omission, not a fact about how the runs were produced: every one used the 4090, the confirmatory bf16 teacher cache (`checkpoints_scratch/teacher_cache/*/cache_key.json` — every D4-D9 cache key reads `precision: bfloat16`, none NF4), the full 15,278-row train split, seed 17, and the same 10-epoch/patience-2/LoRA-rank-16 policy every confirmatory B3/B5/X2 row used. Corrected the stored `pilot` field to `false` in each run's `resource_usage.json` and `recipe_config.json`, recording the correction (old value, reasoning, date) inside `resource_usage.json`'s own `corrections` list rather than silently overwriting it — the same pattern already used for the G1 `freeze_release.py` hash correction (2026-09-06 row above). Then, for each adapter, re-ran inference over the frozen val split (`evaluation/zero_shot_inference.py --adapter ...`) and registered the result through `record_run.py --confirmatory`. **First attempt was wrong and was caught before recording, not after**: D4/D5/D6/D9 are two-stage rows whose S2 adapter was trained by `train_kd.py` on top of the stage-F/-P parent adapter *merged into the base weights* (`if config.parent_checkpoint: ... model.merge_and_unload()`, `distillation/train_kd.py`); evaluating the S2 adapter alone against the raw pretrained base — what `zero_shot_inference.py --adapter` did on its own — silently drops the stage-one alignment entirely. D4 scored 43.1% this way, 8.5 points below its own training-time best_val_macro (51.59%); the same gap appeared on D5. Both wrong records were deleted before being left in `INDEX.md` (they had reached `record_run.py`, not further). Fixed by adding `--parent-adapter` to `evaluation/zero_shot_inference.py`, merging it into the base before the S2 `--adapter`, mirroring `train_kd.py`'s own composition exactly. Re-run, every row now reproduces its training-time best_val_macro within ~0.5 points (D4 51.5%, D5 49.6%, D6 53.1%, D9 47.7%, D7 56.4% vs. training's 51.59/50.13/53.50/47.88/56.82 — greedy-decode-level noise, not a discrepancy). D7 needed no fix: `stage="joint"` has `parent_checkpoint=None`, so its single adapter was never affected. Each run got a run id, `manifest.json`, `metrics.json`/`metrics_constrained.json`, and an `INDEX.md` row. `distillation/runner.py`'s D9 declaration (stale "not run" notes/comment, predating the 2026-09-08 run) updated to match. Refreshed §11's G5-G8 rows, which still described a three-seed condition superseded 2026-09-07 and blank "❌ open" statuses that no longer reflected what artifacts existed or didn't. **This is bookkeeping and inference only — no new training was run, and no test-split number exists yet.** | Author instruction: get these ready for the test-scoring/lock step, run in parallel with the doc refresh | No — every number touched is val, already seen; no test row exists |
| 2026-09-09 | §8, §11 | **Primary/reported recipe decided: D7**, not D6. D6 was §8.2's "method as submitted" (P → S2: feature alignment + small raw KD, then CE + LoCa KD) at 53.50% val; D7 (joint feature + CE + LoCa KD, single pass, `stage="joint"`) scores 56.82% val, a 3.3-point gap, one seed each. D7 uses the joint single-pass shape §13.1 point 5 rejected for D3/D5/D8 — that rejection was about isolating those rows' stage contributions, not a blanket rejection of joint training, and D7 was deliberately kept available as its own row for exactly this comparison (2026-09-07 changelog row above). §8's recipe table annotations swapped accordingly. This decision was made on **val** evidence only, before any test-split number exists — consistent with §5's lock-before-test rule, since choosing the reported recipe is part of what "settings locked" means. | Author decision ("okay let's pick D7 because that makes sense") | No — val only, no test row exists at time of decision |
| 2026-09-09 | §5 | **Settings locked**, satisfying §5's precondition for touching test ("Test evaluation: Once ... after settings are locked"). Locked: primary/reported recipe **D7** (row above); comparator **B3** (§4's "stronger of B3 and B4" resolves to B3 by elimination — B4 was declared in `recipe_library()` but never run, and dropping it is not itself a new decision, just a standing fact restated here for the record); teacher precision **bf16** (§13.1 pt 2, resolved 2026-09-06); seed **17** (§5, single-seed since 2026-09-07); prompt style **terse**, decoding **greedy, `max_new_tokens=16`, `enable_thinking=False`** (§8.3/§2); LoRA rank **16**, epoch/patience policy **10 max / patience 2** (§7.3). Nothing above changed to produce this row — it restates decisions already made and dated; this row exists because §5 wants the lock itself stated, not only inferable from accumulated decisions. | Author instruction ("do all the steps"), precondition for the test-scoring pass this date | No — no test row exists yet |
| 2026-09-09 | §13.1 pt 4 | **B6 (RGB-D reference) dropped for this submission.** Needs a two-image VLM input path neither `train_student.py` nor `evaluation/zero_shot_inference.py` has; already scoped as optional ("a contextual baseline, not a matched competitor"). Not worth building new input-handling code to run a single non-competing reference against the remaining test-scoring/D2/D8 work. | Author instruction ("do all the steps"), resolving a `§13.1` item left open since 2026-09-05 | No |
| 2026-09-09 | §8.2, §9.2 | **`stage_one()`'s default `trainable_modules` reverted from `(vision_attention, vision_merger)` to `(vision_attention,)`; `D0`'s declaration in `recipe_library()` likewise.** Found while preparing D8's stage-F reuse for training: `stage_one_p()` (D6's **P** stage) correctly still trains the merger — D6's real `stage_P` run used it and worked (best-reading 47.91%, feeding D6's real 53.1% test-eligible result) — but `stage_one()` (the plain **F** stage D0/D3/D5/D8 derive from) had kept the same merger-inclusive default even though the *feature-only* variant of it was measured and rejected as sub-chance the same day it was added (`align_curve_contrastive_merger_s17`, 25.84% -> 22.92%, 2026-09-07 changelog). D4/D5/D9 never hit this in practice because their real parent (`align_curve_contrastive_s17`) predates the merger change and was reused via `--parent-checkpoint` rather than regenerated — so the bug was latent, not yet triggered: a fresh call to `stage_one()` today (e.g. to build D8's own F stage, had `align_curve_cosine_s17` not already existed as a manually preserved attention-only run) would have silently reproduced the rejected configuration. Reverted rather than left as a documented gotcha, since the fix costs nothing (no retraining — D4/D5/D6/D7/D9's already-recorded numbers never went through this code path) and a live footgun in the actual recipe-derivation code is worse than a note about one. `tests/test_features.py`'s matching assertion updated; full suite still 257 passed. | Found while sourcing D8's stage-F parent for training (author instruction, "do all the steps") | No |
| 2026-09-09/10 | §5, §11 | **Test split scored, once, for the primary set — G5 closes.** `evaluation/paired_bootstrap.py` written (paired cluster bootstrap by scene group `sequence_id`, 10,000 replicates, seed 20260905, full macro recomputed inside every draw, exactly as §5 predeclares — nothing was decided here, only implemented, and validated against val first where a wrong result costs nothing). Test predictions generated the same way as the val re-recording (`zero_shot_inference.py`, `--parent-adapter` composition for the two-stage rows) and registered via `record_run.py --confirmatory --split test` for B3, B5, X2, D4, D5, D6, D9, D7, in that order (D7 last, deliberately no different treatment than the rest). Results: B3 (primary comparator) 43.6%, B5 59.9%, X2 44.6%, D4 50.6%, D5 50.3%, D6 52.6%, D9 48.4%, **D7 (primary/reported recipe) 54.1%**. **D7 vs. B3: +10.5 points, 95% CI [+9.6, +11.4], excludes zero in all 10,000 replicates** — clears §5.1's threshold by a wide margin (`runs/kd/confirmatory_recording/D7_vs_B3_test_bootstrap.json`). Secondary pairwise CIs vs. B3 (all `runs/kd/confirmatory_recording/*_vs_B3_test_bootstrap.json`): D4 +7.0 [6.2,7.9], D5 +6.7 [5.8,7.6], D6 +9.0 [8.2,9.9], D9 +4.8 [4.0,5.7], X2 +1.0 [0.5,1.5] (statistically real but below the practical 2-point bar). D4-vs-D5 (raw vs. LoCa KD, same stage-F parent): +0.3 points, CI [-0.2,+0.8], **includes zero** — see G6 for what this does and does not resolve. D2/D8 will be test-scored in a follow-up pass once trained; this does not reopen the numbers or the D7 decision recorded here, since D2/D8's design was fixed before either had a test score (2026-09-09 "Settings locked" row). | Author instruction ("do all the steps") | Yes for this row's own content — it reports the test result the row exists to record. No prior decision in this protocol was made after seeing it: recipe (D7), comparator (B3), and the uncertainty method were all locked beforehand (2026-09-09 rows above) |

### 13.1 Open decisions awaiting author sign-off

1. ~~**§9.5 teacher precision**~~ — **RESOLVED 2026-09-05: Option A.**
2. ~~**Expected 4090 return date**~~ — **RESOLVED: the 4090 is online as of 2026-09-05**,
   ahead of the 2026-09-06/07 estimate. Per Option A, confirmatory teacher caching
   and all three-seed runs now proceed on this card at a single locked precision.
   Weight-only arithmetic (§9.4) suggested bf16 fits the 9B/8B teachers with headroom
   (~19.3 GB / ~17.1 GB against 24 GB). **Single-example measurement, 2026-09-05:**
   `Qwen/Qwen3.5-9B` loads bf16 (no quantization) and runs one generate() call on this
   4090 at **18.93 GB peak allocated** (`enable_thinking=False` rendered correctly,
   matching `pilot_findings.md` §7) — consistent with the arithmetic estimate and
   leaving ~5 GB headroom. **RESOLVED 2026-09-06 — Gate G4 measurement taken:**
   `distillation/build_teacher_cache.py` (real batch-4 RGB depth-substitution
   batches, teacher-forced through the actual caching pipeline, top-K=4096
   extraction included) measured **20.61 GB peak** and **7.53 examples/second**
   sustained. This supersedes the single-call datapoint above with the Phase 5
   profiling measurement §13.1 point 2 originally asked for. ~3.9 GB headroom
   remains against the 24 GB card, above the §9.4 ≈2 GB target. **bf16 is
   confirmed as the frozen confirmatory teacher precision**, not merely the
   leading candidate. At 7.53 ex/s, caching the full 15,278-row train split is
   ≈34 minutes, one-time.
3. **Target venue** — deliberately unset; `NEW_SUBMISSION.md` §17 records that
   BMVC 2026 and WACV 2027 deadlines have passed. Choose after Gate G5.
4. ~~**Whether B6 (RGB-D reference) is retained**~~ — **RESOLVED 2026-09-09: dropped for this submission.** It needs a two-image VLM input path that does not exist in `train_student.py`/`evaluation/zero_shot_inference.py` (both build a single image per row) — a new feature, not a training run, and B6 was already described here as "a contextual baseline, not a matched competitor," i.e. optional. Building two-image input to run a single contextual reference is not justified against the remaining work that *does* gate submission (test scoring, D2/D8). Dropped per §7's rule ("Dropping is recorded in §13 with the reason — silent omission is prohibited"), not silently omitted. Retained as future work if the study is extended.

5. ~~**D3/D5/D8 encode a decorative feature term**~~ — **RESOLVED 2026-09-07: option (a), two stages** (author sign-off). Implemented; see the §13 row of the same date. The original finding, kept because it is the reason the row shapes changed: measured,
   not inferred: those three rows declare `stage="S2"` with a feature objective
   and the default `("language_attention",)` surface. A pooled *vision* feature
   loss has no gradient path to language attention, so their feature term
   updates only the alignment head and **contributes nothing to the student** —
   the same silent no-op that D0 had before its surface was corrected (0 of 48
   LoRA tensors receiving gradient). Their CE and KD terms still train normally,
   so the row would produce a plausible result while its distinguishing
   ingredient did nothing. Two coherent resolutions, and the protocol already
   points at one:

   * **(a) Two-stage, recommended.** §7.1 already describes D3–D9 as two-stage
     rows that "pay that cost roughly twice, since stage one is a separate
     training pass before S2 begins," and §8.2 already says S2 freezes the
     vision encoder. On that reading the feature objective belongs to each
     row's **F stage** (which is what D0 is), and its S2 stage is CE/KD only,
     started from the F checkpoint via `RecipeConfig.parent_checkpoint`. This
     also matches the author's own two-phase method. It requires the F->S2
     chaining orchestration, which is not built yet, and removing
     `feature_objective` from the S2 configs.
   * **(b) Joint, single-pass.** Give those rows
     `("language_attention", "vision_attention")` and train the feature and
     answer objectives together. This is a different method from the two-phase
     one and contradicts §8.2's S2 definition ("vision encoder frozen"), so it
     needs a §8.2 amendment rather than only a config change.

   Nothing should run these three rows until this is settled: under (a) they are
   two-stage rows whose current definition is wrong, and under (b) they need a
   surface they do not currently declare. D0 is unaffected and is runnable now.

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
