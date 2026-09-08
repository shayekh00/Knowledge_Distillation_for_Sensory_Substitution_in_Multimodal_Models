# Pilot findings

**Status:** running record of what the 16 GB pilot has established
**Started:** 2026-09-05
**Hardware:** RTX 4080 SUPER, 16 GB (the 4090 is temporarily unavailable)
**Data layer:** [`runs/INDEX.md`](../../runs/INDEX.md) — every number here links to a run id

> **Everything below is PILOT** under Option A of [`experiment_protocol.md`](experiment_protocol.md)
> §9.5. These runs establish feasibility, calibrate settings, and catch defects.
> No number here may appear in a main or ablation table.

This document is the *interpretation* layer. `runs/INDEX.md` and the per-run
directories are the data layer, and they are generated, never hand-edited. Where
the two disagree, the run directory is right.

---

## 1. Feasibility: the 16 GB card is not the constraint

| Workload | Peak VRAM | Throughput |
|---|---:|---:|
| Zero-shot inference, Qwen3.5-0.8B bf16 | **1.80 GB** | 9.9 it/s |
| LoRA training, r16, gradient checkpointing | **3.27 GB** | ~2 ex/s |

A full training epoch over the 15,278 train rows takes **101.9 minutes**, measured
(B3, 0 rows skipped). The card has roughly 4x headroom over what LoRA training
actually uses, and inference and training were observed running *concurrently*
without exhausting it.

> **Superseded 2026-09-06 — the paragraph below was wrong.** It is kept because
> it was acted on, and because the way it was wrong is instructive. See §10.

~~**The bottleneck is the data path, not the GPU.** Utilisation sat at ~19% during
training: depth decoding, the Prewitt pass, and image preprocessing all run inline
and single-threaded. Moving them to a `DataLoader` with workers should give a
3-5x speedup — plausibly 30 minutes per epoch rather than 100.~~ Measured
directly on 2026-09-06: the data path is **1.7%** of a training step and runs at
~120 ex/s in isolation, against 2.1 ex/s for the step as a whole. A `DataLoader`
could therefore never have returned more than 1.7%. The ~19% figure was read as
"the GPU is idle waiting for data", but `nvidia-smi` utilisation reports only
whether *some* kernel is resident, which at batch size 1 with gradient
checkpointing says nothing about efficiency. What actually bound throughput was
batch size and gradient checkpointing; §10 records the corrected numbers and the
6.7x that fixing them delivered.

The earlier assumption that Phases 6–12 were blocked on the 4090 was wrong. What
actually binds is **wall-clock, not memory**: one-seed pilots are comfortable, the
39-config-seed confirmatory core is weeks. That is the real argument for waiting,
and it is an argument about time.

Feeds Gate **G4**, which asks for measured throughput rather than arithmetic.

## 2. The §6.4 prompt is broken for a small student

The single most consequential pilot result so far.

Identical model, identical inputs, identical greedy decoding. **Only the
instruction wording differs.**

| Run | Prompt | Macro | Invalid |
|---|---|---:|---:|
| [`…-B1-s17-e20002c9`](../../runs/pilot/) | `terse` | **36.4%** | 12.4% |
| [`…-B1e-s17-8355a1d5`](../../runs/pilot/) | `enumerated` (§6.4) | **23.0%** | **59.7%** |

Per type, under the §6.4 wording:

| Type | accuracy | invalid |
|---|---:|---:|
| existence | 56.8% | 1.4% |
| left_right | 56.5% | 0.0% |
| identify_superlative | **0.0%** | **100.0%** |
| nearest_object | **0.0%** | **100.0%** |
| relative_depth | 1.7% | 98.0% |

The §6.4 instruction is *"Answer with only the short answer: yes, no, left, right,
or the object name, as appropriate."* The 0.8B model **echoes that list back**:

```
val_000000 → "yes, no, left, right, or the object name"
```

The diagnosis is precise. The enumeration is mildly *helpful* for the two binary
types — the legal answers are literally in the prompt, and both score slightly
above their `terse` counterparts. It is **catastrophic** for the three types whose
answers are object names, which collapse to 98–100% invalid and 0% accuracy.

At 23.0% macro, the §6.4 prompt puts the model **below the 30.3% chance floor**.

### Why this was nearly missed

Without the invalid-output column, 23.0% reads as "the model is bad at depth". It
is not: the model is not answering the question at all. That column exists because
of the Phase 3 evaluator repair, and it earned itself here on the first real run.

### What must happen

§6.4 requires **one** instruction for every model, so the wording cannot be chosen
per model. `terse` rescues the 0.8B student — and **§6 below confirms it is also
the better wording for the 9B teacher**, so a single instruction serves both and
the one-instruction rule is satisfiable.

Remaining steps before the prompt is frozen:

1. ~~Run the same three-way comparison on the teacher.~~ Done — see §6.
2. Choose on validation, then freeze — §6.4 forbids retrofitting after test
   results are seen. Doing this now, pre-freeze, on `val`, is the legitimate
   moment.
3. Consider whether constrained decoding (§6.3, `evaluation/candidate_scoring.py`)
   makes the choice less load-bearing. It should: a token trie makes an illegal
   answer unreachable regardless of how the instruction is worded, which would
   remove the size-dependence documented in §6 entirely.

## 3. Zero-shot references

| Run | Modality | Macro | Invalid |
|---|---|---:|---:|
| B1 | depth | 36.4% | 12.4% |
| B2 | RGB | 43.0% | 15.3% |
| — | val baselines | chance 30.3 · random 29.9 · majority 33.5 · **question-only 34.0** | — |

Two readings:

**The zero-shot depth student barely uses the image.** 36.4% against a 34.0%
question-only baseline is a 2.4-point margin. Most of what the untrained model
scores is language prior, which is precisely the headroom supervised training and
distillation have to fill.

**The RGB→depth gap is 6.6 points** for the same model on the same scenes and
questions. That is the sensory-substitution gap the paper exists to close,
measured rather than assumed.

Per type, B1 splits sharply: the binary types sit at 54–55%, barely above their
50% floor, while `identify_superlative` (11.4%) and `nearest_object` (6.4%) are
near the floor of their much larger answer spaces. Whatever the untrained model is
doing, it is not metric-depth reasoning.

## 4. Supervised fine-tuning works, and fixes the format problem outright

**B3** — depth CE, LoRA r16 on `q/k/v/o_proj`, lr 1e-5, effective batch 16, one
epoch over all 15,278 train rows, 954 optimizer steps, 101.9 minutes, 0 rows
skipped, peak 3.33 GB.

| Type | B1 zero-shot | B3 fine-tuned | Δ |
|---|---:|---:|---:|
| existence | 54.5% | 56.0% | +1.5 |
| identify_superlative | 11.4% | 16.4% | +5.0 |
| left_right | 55.0% | 58.8% | +3.8 |
| nearest_object | 6.4% | 12.5% | +6.1 |
| relative_depth | 54.4% | **60.2%** | +5.8 |
| **Macro** | **36.4%** | **40.8%** | **+4.4** |
| **Invalid** | **12.4%** | **0.0%** | **−12.4** |

Four things worth drawing out.

**Every type improved**, and the largest gains are on the three depth-relation
types — the ones that actually require metric depth rather than 2-D layout or a
language prior.

**`relative_depth` reaches 60.2%**, ten points clear of its 50% chance floor.
That is the strongest evidence so far that the student is doing something with
depth rather than with priors.

**Invalid outputs fall to 0.0% on every type.** The model learns the answer
format completely. This bears directly on §2 above: the §6.4 prompt crisis is a
*zero-shot* problem. For trained rows the instruction wording matters far less,
because the format is learned from the targets. The prompt still has to be fixed
— B1/B2 are reported rows — but it is not a threat to the trained comparisons.

**A fine-tuned depth student (40.8%) now sits within 2.2 points of a zero-shot
RGB model (43.0%)**, which is a compact statement of what the paper is about.

### What this row is not

One epoch, one seed, one learning rate. §4 of the protocol requires the primary
comparator to get a *fair tuning budget* — three learning rates, validation
selection — before any KD result is measured against it. B3 as it stands is a
first pass that establishes the pipeline works and roughly where CE lands. It is
**not yet** the strong CE baseline the paper compares against, and no KD number
should be placed beside it until it is.

## 5. Teacher suitability: the KD premise holds

§10.3 flags a precondition of the entire study — *"A large RGB teacher may be
weaker than a depth student on measured depth relations."* If that were true here,
the premise would need revisiting before any pipeline was built. **It is not true.**

All runs below: val split, PILOT. Cells are accuracy / invalid-output rate.

| Run | Macro | Invalid | existence | left_right | relative_depth | identify_sup | nearest_obj |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 student depth 0-shot | 36.4% | 12.4% | 54.5% | 55.0% | 54.4% /11% | 11.4% /40% | 6.4% /11% |
| B2 student RGB 0-shot | 43.0% | 15.3% | 73.0% | 47.6% /24% | 57.6% /15% | 22.0% /25% | 14.6% /13% |
| **B3 student depth fine-tuned** | **40.8%** | **0.0%** | 56.0% | 58.8% | 60.2% | 16.4% | 12.5% |
| **T1 teacher depth 0-shot** | **44.7%** | 3.7% | 59.4% | 61.2% | 70.1% | 17.6% /11% | 15.5% /8% |
| **T2 teacher RGB 0-shot** | **62.2%** | 4.5% | **84.9%** | **89.7%** | **78.8%** | **29.3%** /12% | **28.3%** /10% |

Teacher: Qwen3.5-9B, NF4, thinking disabled. Student: Qwen3.5-0.8B, bf16.

### The teacher beats the fine-tuned depth student on every type

| Type | B3 fine-tuned depth | T2 teacher RGB | Teacher advantage |
|---|---:|---:|---:|
| existence | 56.0% | 84.9% | **+28.9** |
| left_right | 58.8% | 89.7% | **+30.9** |
| relative_depth | 60.2% | 78.8% | **+18.6** |
| identify_superlative | 16.4% | 29.3% | **+12.9** |
| nearest_object | 12.5% | 28.3% | **+15.8** |

Crucially the advantage holds on the **three depth-relation types**, which is
exactly where §10.3 warned it might not. The teacher has substantial, transferable
signal on the questions the paper is about. There is something to distil.

### Two further readings

**The teacher zero-shot on depth (44.7%) already beats the fine-tuned student
(40.8%).** Scale alone, with no target-task training, exceeds a full epoch of
supervised adaptation on a 0.8B model. The small student has a lot of headroom
left, and B3's one-epoch/one-LR configuration is clearly not its ceiling.

**Modality costs the teacher more than the student.** Switching RGB → depth costs
the teacher 17.5 points (62.2 → 44.7) but the student only 6.6 (43.0 → 36.4) —
because the teacher has far more RGB capability to lose. That gap *is* the
sensory-substitution problem, stated quantitatively: 17.5 points of capability
that currently evaporate when the sensor changes.

`relative_depth` is the encouraging one: 78.8% for the teacher on RGB and 70.1%
on depth, both far above the 50% floor. Metric-depth relations are learnable from
this benchmark, not noise.

### Caveats

NF4 quantization and one seed. The teacher's numbers are quantization-specific; a
bf16 teacher on the 4090 may differ, and these are PILOT rows either way.

## 6. The prompt question is resolved: one wording serves both models

§6.4 requires a single instruction for every model, so the `terse` fix could not
be adopted on the student's evidence alone. Running the teacher both ways settles it:

| Model | terse | enumerated (§6.4) | Δ macro | Δ invalid |
|---|---:|---:|---:|---:|
| Student 0.8B (B1 / B1e) | 36.4% | 23.0% | **−13.4** | 12.4% → **59.7%** |
| Teacher 9B (T2 / T2e) | 62.2% | 58.5% | −3.7 | 4.5% → 9.4% |

The enumerated wording hurts **both** models, so `terse` is the better instruction
for the pair and no per-model tuning is needed — §6.4's one-instruction rule is
satisfiable. The magnitude differs enormously, though: the 9B shrugs it off
(−3.7 points), while the 0.8B collapses below chance with 60% invalid outputs.
Prompt robustness is strongly size-dependent, which is worth a sentence in the
paper: a wording validated only on a large model can be catastrophic for the
small one that actually gets deployed.

## 7. The teacher is a reasoning model, and its chat-template default differs from the student's

A §8.1.1 compatibility-gate finding that would have silently corrupted every
teacher cache.

Qwen3.5's chat template supports `enable_thinking` and emits `<think>` blocks.
Run with template defaults and a 16-token budget, the 9B teacher produces
**truncated chain-of-thought instead of answers**:

```
val_000000 → "The user is asking a simple yes/no question about the presence of a lamp in"
val_000001 → "The user wants to know which object is farther away: the picture frame or the"
```

With `enable_thinking=False` the same model, inputs and decoding give:

```
val_000000 → "yes"          val_000003 → "left"
val_000001 → "picture frame" val_000004 → "monitor"
val_000002 → "light"         val_000005 → "bed"
```

**The two checkpoints default differently**, which is the part worth recording:

| Model | Rendered assistant prefix (default) | Effect |
|---|---|---|
| Qwen3.5-0.8B | `<think>\n\n</think>\n\n` — **closed** empty block | thinking already off |
| Qwen3.5-9B | `<think>\n` — **left open** | model generates reasoning |

So the student was always effectively thinking-off, and **B1/B2/B3 are valid and
matched as recorded** — they need no re-run. Setting `enable_thinking=False` on
the teacher renders the *identical* prefix to the student's default, so the fix
brings the pair into alignment rather than introducing a decoding mismatch under
§9.3.

### Why this matters beyond the pilot

Had the teacher been cached without catching this, every cached target would have
been a chain-of-thought token distribution rather than an answer distribution —
and the KD runs would have distilled the teacher's *narration of the task*. The
resulting numbers would have been low, plausible, and completely
uninterpretable: exactly the failure mode §4.1 records for the historical
all-zero KD tables.

**Requirement.** The decoding contract (§6.4) must specify `enable_thinking=False`
explicitly for every model, not rely on template defaults, and the compatibility
gate must render and inspect the actual prompt string for each checkpoint rather
than assuming family members agree.

## 8. Environment findings

**The pinned `transformers` cannot load the portfolio.** 4.49.0.dev0 has no
`Qwen3_5ForConditionalGeneration`; Qwen3.5 requires **≥ 4.57**. Also missing
`libgl1` and `torchvision` for the Qwen3.5 processor. Resolved with a separate
`.venv-models` (transformers 5.16.1) so the environment the 207 tests run against
stays untouched. `requirements.txt` cannot serve both the legacy and new
portfolios.

**The cache arithmetic was optimistic.** Qwen3.5's vocabulary is **248,320**, not
the 150k assumed in §8.2. Dense caching at 16 answer positions is ~121 GB, not
73 GB; top-K at 4096 is ~6 GB once int32 token ids are counted alongside scores.
`NEW_SUBMISSION.md` §8.2 has been corrected — the plan asked for this to be
measured in the pilot, and it now is.

## 9. The 4090 is online: first real measurement confirms the data-path diagnosis

**2026-09-05, RTX 4090 24GB.** `distillation/train_student.py --recipe B3 --limit 300`
(same recipe as §4's B3: LoRA r16 on q/k/v/o_proj, lr 1e-5, effective batch 16,
depth modality, replicated representation), fresh environment, real SUN RGB-D
images (the earlier truncated download was fixed first).

| Card | Peak VRAM | Throughput | Projected full epoch (15,278 rows) |
|---|---:|---:|---:|
| RTX 4080 SUPER, 16 GB (§1) | 3.27 GB | ~2 ex/s | 101.9 min (measured) |
| RTX 4090, 24 GB (this run, 300 rows) | **3.33 GB** | **2.17 ex/s** | 117.4 min (projected) |

VRAM is essentially unchanged, as expected — the student and its LoRA adapter are
tiny relative to either card. The load-bearing result is throughput: **the 4090
is not meaningfully faster than the 4080 SUPER on this workload.** Loss fell
2.23 → 1.47 over 18 optimizer steps, confirming the training loop itself runs
correctly end to end on this machine.

> **Corrected 2026-09-06.** This section originally read the equal throughput as
> confirming §1's data-path diagnosis. That inference was wrong, and wrong in a
> tempting way: two cards agreeing does rule out *raw GPU capability* as the
> binding constraint, but it does not identify the data path as the cause. The
> shared cause was the batch-1, gradient-checkpointed configuration, which both
> cards ran identically. §10 has the direct measurement.

This is a 300-row, one-seed smoke measurement (`runs/pilot/B3_depth_4090_profile/`,
PILOT, not a Gate-G4-closing run) — it establishes that the pipeline works and
roughly re-confirms throughput, not the full §8.5 profiling protocol (20 warm-up +
100 timed steps per phase, teacher caching, resume cycle).

Separately, `Qwen/Qwen3.5-9B` (the primary teacher) loads bf16 with no quantization
at **18.93 GB peak** on this card (single forward+generate call, blank image) —
consistent with the arithmetic estimate in §8.2/§9.4 and leaving ~5 GB headroom.
`google/gemma-4-12B-it` loads bf16 at **24.18 GB peak**, i.e. at the card's limit
with no headroom, confirming it needs quantization here exactly as §9.4 anticipated.

---

## 10. What actually bound throughput: batch size, not the data path

**2026-09-06, RTX 4090.** §1 attributed the 2 ex/s ceiling to the inline
depth-decode/Prewitt path and predicted a 3-5x win from DataLoader workers. That
was tested and is false. Three measurements, in the order they were taken:

| Measurement | Result |
|---|---|
| Data path alone, 300 rows, no GPU | ~120-160 ex/s |
| Training step split with `cuda.synchronize()` | data **1.7%** / fwd+bwd **98.3%** |
| Gradient checkpointing off, batch 1 | 2.11 → **3.91 ex/s** (1.85x), 3.09 → 5.37 GB |
| Batch 4, no checkpointing | **12.84 ex/s** (6.1x over batch 1) |

The DataLoader could never have paid more than 1.7%. Implementing it also
surfaced an unrelated environment limit worth recording: this container's
`/dev/shm` is **64 MB** while one example's `pixel_values` is 5-10 MB, so torch's
worker→parent tensor transport deadlocks (`No space left on device`) after a few
prefetched items. Anything multi-process and tensor-passing on this machine needs
`--shm-size` raised; the DataLoader code itself was reverted as pointless.

**Why §1 misread it.** `nvidia-smi` utilisation reports whether a kernel is
resident, not whether the device is doing useful work. At batch 1 with gradient
checkpointing the kernels are tiny and launch-latency-bound, so ~19% meant
"inefficient", not "starved". The lesson is that utilisation is not a bottleneck
attribution; splitting the step and synchronising is.

**Adopted configuration:** batch 4, gradient checkpointing off. Full epoch
**21.1 min at 12.1 ex/s, peak 14.6 GB** (worst case — largest `kv2` frames with
the longest questions — 13.66 GB, seq 449), against 141 min before. **6.7x.**

### 10.1 CE is now normalised per example, and batch size is inert

Batching initially pooled loss across the batch, which weights a row by its
answer token count and would have made a KD row that shrinks its batch to fit the
teacher cache incomparable to CE. `masked_cross_entropy` now averages within a
row and then across rows, so batch 4 with accumulation 4 optimises the same
objective as batch 1 with accumulation 16 — pinned by
`test_ce_is_invariant_to_micro_batch_size`, and confirmed end to end on real data
(B3 three-seed: 42.54% at batch 1 versus 42.58% at batch 4, inside seed noise).

This also aligns the objective with the endpoint: macro accuracy counts a
question once whether its answer is "no" or "file cabinet", whereas token pooling
quietly upweighted the open-vocabulary types. Gathering only supervised positions
before the float32 upcast additionally cut ~3.5 GB per step, since the upcast no
longer spans every padded position of a 248,320-wide vocabulary.

### 10.2 Baselines at the adopted configuration (val, `terse`, lr 2e-5)

> **Replaced 2026-09-07.** The figures below are the re-run under both the
> §10.4 `<think>` fix and §7.3's 10-epoch/patience-2 policy. The table this
> replaced held single-epoch numbers trained under the prompt defect; its
> B3 figure (42.58% +/- 0.51, 3 seeds) understated CE by ~2.3 points and should
> not be quoted anywhere. Zero-shot rows were never affected.

All trained rows: seed 17, `terse`, lr 2e-5, LoRA r16 on attention projections,
batch 4 / effective 16, up to 10 epochs with patience 2, **best-epoch**
checkpoint (not last), scored on the full 1,720-row val split.

| Row | Macro | Best epoch | Epochs run | Note |
|---|---:|---:|---:|---|
| B1 zero-shot depth | 36.18% | — | — | inference only |
| B2 zero-shot RGB | 42.61% | — | — | inference only |
| **B3 depth CE** | **44.9%** | 2 | 5 | **primary comparator** |
| X2 depth CE + X-Token KD | 44.4% | 3 | 6 | teacher Qwen3.5-9B, cache `0462bee1d6f00444` |
| B5 RGB CE | 62.2% | 6 | 9 | contextual reference |
| B3 depth CE, seed 42 | 44.0% | 2 | 5 | seed-spread anchor only (§5 is single-seed) |

**The headline: X-Token KD does not beat CE.** At a matched training budget it
lands **0.5 points below** the CE baseline (44.4% vs 44.9%). §5.1's success
threshold is a **>= 2-point** gain with a paired interval excluding zero, so this
is not a marginal miss — no KD benefit is demonstrated. The earlier
encouraging-looking result (X2 42.5% vs B3 42.13%, "+0.4 for KD") was an artifact
of comparing two *undertrained* models: both had trained a fixed 1 epoch, which
was the CLI default rather than a validated choice (§13, 2026-09-06). Giving both
arms the schedule §7.3 always specified moved CE up 2.8 points and KD up 1.9, and
reversed the sign of the difference.

Three cautions on reading that:

1. **The seed spread exceeds the effect.** B3 run twice under this policy gives
   44.9% (seed 17) and 44.0% (seed 42) — a 0.9-point spread from seed alone,
   larger than the 0.5-point CE-vs-KD gap. Sub-point differences in this study
   are therefore not interpretable, and under the single-seed policy adopted
   2026-09-07 (§5) they will not become so. This does not rescue KD: a 0.5-point
   gap fails a >= 2-point threshold at any seed count. It does mean the honest
   claim is *"no measurable KD benefit,"* not *"KD is worse."*
2. **KD's loss was learning; it just did not pay.** X2's KD term fell steadily
   (0.54 -> 0.48) while its val macro plateaued. The distillation signal is being
   fit — this is not a wiring failure of the kind §7.3 warns about.
3. **In-training and re-scored numbers differ by ~0.15 points** (B3 45.03 vs
   44.9; X2 44.56 vs 44.4 — 2-3 questions of 1,720). `epoch_loop.py` claims to
   match `zero_shot_inference.py` decoding exactly, so this is an unexplained
   gap, probably batched-generation padding. It is small, but the early-stopping
   decision is made on the in-training number, so it is on the Open list.

The LR sweep (5e-6: 40.72%, 1e-5: 40.95%, 2e-5: 42.81%) selected **2e-5**, about
1.7 points above the pilot's 1e-5 choice — the comparator KD must beat is
correspondingly stronger. Because these are val numbers and 2e-5 was chosen on
val, they carry selection optimism by construction; test is scored once after
G5 locks.

Read against the pilot's teacher rows (T1 depth 44.7%, T2 RGB 62.2%, both NF4),
the picture sharpened when B5 was re-run: the 0.8B student on RGB now scores
**62.2%, matching the 9B teacher's RGB number outright** (was "within 2.5 points"
when B5 stood at 59.69%). So **the ~17-point depth deficit is a modality limit,
not a capacity limit** — the premise the study needs, now held more firmly than
before.

But the same update sharpens the problem for KD. The teacher's advantage on
*depth* is only ~2 points (44.7% vs the student's 44.9% — the fine-tuned student
has now drawn level with, or passed, the NF4 teacher on depth). A distillation
signal has to come from the teacher's RGB view, and X2 reads that view only
through cached top-K logits over answer tokens. That the student already matches
the teacher on RGB and equals it on depth is a plausible mechanical reason X2
gained nothing: at a matched budget there may be little the teacher's answer
distribution knows that CE on gold labels does not already supply. This is a
hypothesis the ladder was built to test, not a settled explanation — it is
exactly what the LoCa and feature-alignment stages (D2/D5/D8) probe, since those
transfer *intermediate* structure rather than answer distributions.

### 10.3 A provenance defect this exposed

Recording the batch-4 runs silently overwrote the batch-1 ones: `record_run.py`
hashed a hardcoded field list that omitted `batch_size`, `effective_batch` and
`gradient_checkpointing`, so two materially different runs shared a run id —
precisely the collision `make_run_id`'s docstring promises cannot happen. Fixed
by hashing the batch schedule from the training script's own manifest, plus a
guard that refuses to overwrite a stored run whose configuration differs and
prints the diff. No data was lost; the batch-1 rows were re-recorded under
distinct ids and kept as the cross-check above.

### 10.4 Training was teacher-forced after an *open* `<think>` tag

§7 found that Qwen3.5's chat template emits chain-of-thought by default and that
inference must pass `enable_thinking=False`. `evaluation/zero_shot_inference.py`
does. **`distillation/train_student.py`'s `build_batch` did not**, and nothing
compared the two, so every CE run in §10.2 trained under a different prompt than
the one it was scored with:

```
build_batch (training):        ...<|im_start|>assistant\n<think>\n
zero_shot_inference.py (eval): ...<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The training prompt leaves the thinking block **open**, so the gold answer was
teacher-forced as though it were the opening tokens of the model's own
reasoning rather than its final answer. Decoding the teacher's top-1 prediction
at each supervised position shows what that meant in practice — the same eight
examples, before and after passing `enable_thinking=False`:

| Gold | Top-1 under the open tag | Top-1 under the closed tag |
|---|---|---|
| `microwave` | `'The'` (p=1.00), `'rowave'`, `'\n'` | `'mic'` (0.84) → `'rowave'` (1.00) → EOS (1.00) |
| `drawer` | `'The'` (p=1.00), `'\n'` | `'drawer'` (0.96) → EOS (1.00) |
| `right` | — | `'right'` (0.93) → EOS (1.00) |

The open-tag column is not a weak signal, it is a different task: the model is
predicting how a reasoning trace would begin. This matters most for KD, where
the cached teacher distribution *is* the target — a cache built this way would
have distilled "how Qwen starts a chain of thought" into the student.

**Fixed** in `build_batch`, which is the single function both `train_student.py`
and `build_teacher_cache.py` call, so the correction reaches CE and KD alike.
Two things it exposed:

* **The prompt hash did not cover the prompt.** `build_teacher_cache.py` hashed
  `PROMPT_SUFFIX`, which is unchanged by `enable_thinking`, so the corrected and
  uncorrected caches collided on one key — the §9.5 failure mode exactly. It now
  hashes the fully rendered chat template.
* **The frozen prompt in `experiment_protocol.md` §8.3 was never the one being
  run.** §8.3 named an answer-space-enumerating wording; every measured number
  in this document used `terse`. Resolved by amendment (protocol §13,
  2026-09-06): `terse` is the frozen prompt, and `zero_shot_inference.py`'s
  default was changed from `enumerated` to match.

The §10.2 numbers were all produced under the defect and are being re-run. First
result: the LR sweep re-run under the fix selects **2e-5 again** (42.1% vs 40.7%
at 5e-6 and 40.4% at 1e-5), and seed 17's corrected score lands close to the
uncorrected three-seed mean. So the defect does not appear to have moved CE's
ceiling much — LoRA can evidently bridge a template-suffix mismatch when the
image and question are unchanged. It still had to be fixed: train/eval identity
is a §4 fairness obligation independent of whether it moves a number, and the
teacher cache is unusable without it.

---

## 11. X-Token verified on real Qwen3.5-9B/0.8B data, including a same-family
    surprise

Protocol §9.3 gives this pair a specific job beyond the KD ladder itself: since
its tokenizer is shared, X-Token's projection must reduce to plain token KL
here, and disagreement would mean a projection bug — the one pair where the
method can be checked against ground truth before it is trusted on pairs
(InternVL, SmolVLM2, Gemma-4) where no such check is possible.

**"Same tokenizer family" turned out not to mean "same integer vocabulary."**
Loading the real saved mapping for Qwen3.5-9B↔0.8B: of 248,077 tokens,
**244,133 have matching ids and 944 (0.38%) do not.** The mismatches are all
byte-fallback tokens — `tokenizer.decode()` on an isolated invalid-UTF-8 byte
returns the same replacement glyph (`�`) regardless of which byte it is, so the
surface-text matching in `build_vocabulary_mapping` cannot tell them apart in
isolation. The most-collapsed case is large: **665 distinct student ids** (a
wide range of raw byte-fallback tokens) all map onto a single teacher id.
Practically low-stakes — an isolated invalid byte essentially never survives
as a final output token, since BPE merges valid multi-byte sequences before
that point — but it is real, and it is exactly the kind of silent assumption
§9.3 already refuses to make ("tokenizer compatibility is not a model-selection
criterion"). It is why `VocabularyMapping.is_identity()` correctly reports
`False` on this pair rather than `True`.

**Verified rather than assumed**, on real data (`distillation/verify_xtoken_identity.py`):

1. **`scatter_add_` accumulates rather than overwrites.** Fed all 665 ids
   collapsing onto the same teacher token equal probability shares (summing to
   1.0); got back **1.000003** (float32 summation noise across 665 terms,
   trivial). Confirms the projection sums every contributing student token's
   mass rather than silently keeping only the last write — the specific way
   this kind of bug would otherwise hide.
2. **The full KD loss matches an independent re-derivation.** A real Qwen3.5-9B
   teacher cache, a real Qwen3.5-0.8B forward pass (6 real training examples,
   14 real supervised positions), `distillation.xtoken.projected_kl_loss`
   (production) vs. the same computation via a Python dict groupby instead of
   `scatter_add_` (independent, different mechanism, not a reformatted copy):
   **1.5571314096 vs 1.5571312870, difference 1.23e-07** — machine-precision
   agreement.

**Conclusion for the paper:** X-Token's projection is correct on the pair whose
job is to prove that, on real model data, including the specific edge case
(many-to-one byte-fallback collapse) that a synthetic fixture would not
surface. This is what licenses trusting the same, unmodified code on the
cross-tokenizer pairs where it is actually load-bearing (§9.3) rather than
merely checkable. It does not by itself validate the mapping quality on a
genuinely distant tokenizer pair (Gemma-4's is the intended stress test,
§9.3) — coverage/exact-match-fraction should be re-checked there before
trusting X-Token's numbers on that pair the same way.

---

## 12. D5 beats CE: the two-stage method clears the §5.1 threshold

Seed 17, 10-epoch/patience-2 policy throughout, full val split every epoch.

**Stage F — contrastive vision alignment** (`align_curve_contrastive_s17`,
`vision_attention` trainable, language model frozen and never run, teacher
feature cache `0462bee1d6f00444`): a 5-epoch fixed budget, full-val reading
curve **34.48 / 35.40 / 35.16 / 34.52 / 33.87%**. `best_reading_epoch=1`
(35.40%) was carried forward as stage S2's `parent_checkpoint` rather than the
policy-kept last epoch, because the curve turns over after epoch 1 and this
number is diagnostic, not selective (§13, 2026-09-07 amendment on the coverage
fix) — the merger-trainable variant of this same stage was tried and rejected
first (below chance floor, same date), so this attention-only curve is the one
that fed D5.

**Stage S2 — CE + X-Token KD + LoCa on the aligned, now-frozen vision tower**
(`language_attention` LoRA r16, merged-and-frozen stage-F adapter as the
starting point): val macro by epoch **46.65 / 48.91 / 49.55 / 49.38 / 49.72 /
50.01 / 49.66 / 50.13%**, patience-stopped after epoch 7, **best = 50.13%**.

| Row | Macro | vs B3 | Note |
|---|---:|---:|---|
| B3 depth CE | 44.9% | — | primary comparator |
| X2 depth CE + X-Token KD (unaligned vision) | 44.4% | −0.5 | §10.2 |
| **D5 depth CE + X-Token + LoCa (aligned vision)** | **50.13%** | **+5.2** | first row to clear §5.1 |

Same seed, same 10-epoch/patience-2 policy, same LoRA rank as B3/X2 — the only
thing D5 changes is the two-stage recipe. **+5.2 points over CE and +5.7 over
plain logit-KD**, comfortably past §5.1's declared ≥2-point bar with room to
spare against the one seed-spread anchor this study has (B3's 44.9%/44.0%
across seeds 17/42, a 0.9-point spread) — 5.2 points is not the kind of gap a
seed swap plausibly erases, unlike X2's 0.5-point miss.

**D3 (CE-only on the aligned vision tower) is intentionally not being run.**
The natural next question is attribution — how much of the +5.2 comes from
stage F's vision alignment versus stage S2's answer-distribution KD — and D3
is the row that would answer it. Author decision, 2026-09-07: leave it
unrun. The manuscript's claim is framed at the level of the whole two-stage
method rather than split by stage: stage F transfers the teacher's visual
feature structure into the student (a distillation target, same as stage S2's
logits, under this project's own §1 definition of what is being distilled), so
the F→S2 pipeline is reported as one knowledge-distillation method and D5's
gain is attributed to that method as a whole, not decomposed into an
alignment-only share and a KD-only share. **Stated plainly as a limitation:**
this means the manuscript cannot say what fraction of the +5.2 points is doing
the work in each stage, only that the combined method produces it, at one
seed. Nothing about the D5 numbers above changes — only that a further
ablation which could answer the attribution question will not be run.

---

## Open

*Updated 2026-09-07 after the multi-epoch CE-vs-KD result (§10.2).*

* ~~**X-Token mapping quality on a genuinely distant tokenizer pair —**~~
  **resolved 2026-09-07.** Built the real Gemma-4-12B-it→Qwen3.5-0.8B mapping
  (`build_vocabulary_mapping`, CPU-only, no GPU needed): **coverage 1.0**
  (every one of 248,077 student tokens maps to something, 0 unmapped), but
  **exact-match fraction only 56.66%** (140,568 exact / 107,509 retokenized) —
  against 99.62% for Qwen↔Qwen (§11), confirming this pair is the real stress
  test §9.3 intended, not a formality. The most-collapsed teacher id absorbs
  **5,880 distinct student ids** (vs. 665 for Qwen↔Qwen). Re-ran §11's
  `scatter_add_`-accumulates check against this mapping: it initially read as
  a failure (0.999972 vs. 1.0, a 2.8e-5 error against the check's `1e-5`
  tolerance), but three independent confirmations show this is float32
  summation noise scaling with the larger collapse, not a projection bug — a
  float64 rerun of the identical call lands at 0.99999999999993 (6.6e-14
  error), and a plain sequential Python float32 sum of 5,880 copies of 1/5,880
  reproduces the *exact same* 0.9999720454216003, bit-for-bit. Fixed
  `check_scatter_add_accumulates`'s tolerance to scale with collapse size
  (`max(1e-5, n_collapsed * 1e-8)`, still 1e-5 on the Qwen-Qwen pair it was
  first tuned on) rather than special-casing this one pair. **Conclusion: the
  projection mechanism itself is confirmed correct on this pair too; the open
  question this leaves is not code correctness but whether a 57%-exact-match,
  266-fold-larger-collapse mapping carries enough signal for X-Token KD to
  help on Gemma-4→Qwen** — an empirical question for whenever that pair is
  actually trained, not a mapping-quality defect.
* **Teacher suitability at the confirmatory precision.** T1/T2 exist only at NF4
  on the 16 GB card; §8.2 warns quantisation moves teacher targets, so bf16
  re-runs are in flight before any cache is built. Still a precondition.
* ~~**Top-K for the teacher cache**~~ — **measured 2026-09-06.** K=4096 confirmed:
  generation runs at **7.53 ex/s, 20.61 GB peak** (bf16, batch 4), so the full
  15,278-row train split caches in ≈34 minutes. Size is far below the earlier
  estimate — **~27 KB/example, ≈410 MB for the split**, because answers average
  only **2.19 supervised positions** and `savez_compressed` roughly halves that
  again. Retained top-K mass is ~1.0 at every position sampled, so 4096 is not
  truncating anything that matters. This measurement closes protocol Gate G4.
* **B4, stage-matched CE.** Deliberately after the KD schedule is fixed, since
  matching that schedule is the row's entire purpose.
* **InternVL3.5 does not load** under transformers 5.16.1 — neither the repo's
  remote code (tied-weights API) nor the library's native `InternVLConfig`, which
  the checkpoint's `internvl_chat` model type does not match. This blocks both
  replication legs; the primary Qwen pair is unaffected. Needs a pinned
  environment, an upstream revision, or a substitute pair.
* **Florence-2** fails separately on `forced_bos_token_id`; screen-only, parked.
* ~~**SmolVLM2-500M — load check pending**~~ — **resolved 2026-09-07:
  loads cleanly.** `AutoModelForImageTextToText.from_pretrained` +
  `AutoProcessor` both succeed, a real image + prompt through
  `apply_chat_template` generates a real answer, **1.49 GB peak VRAM**. This
  is the cheapest model in the portfolio by a wide margin and unblocks the
  cross-family **student** track (Reviewer R2.1) — InternVL3.5 and Florence-2
  were the two that failed to load; SmolVLM2 is not affected by either defect.
* **`/dev/shm` is 64 MB**, which forecloses any multi-process tensor passing
  (§10). Raising it needs a container restart.
* **`epoch_loop` and `zero_shot_inference` disagree by ~0.15 points.** Both
  claim identical decoding, but B3 scored 45.03% in training and 44.9% re-scored
  from its saved checkpoint (X2: 44.56 vs 44.4) — 2–3 questions of 1,720. The
  early-stopping decision and the kept checkpoint are chosen on the in-training
  number, so the two paths need to be reconciled, most likely a batched-
  generation padding difference. Small, but it selects checkpoints.
* ~~**No KD variant has beaten CE yet.**~~ — **resolved 2026-09-07: D5**
  (two-stage contrastive vision-alignment → CE + X-Token + LoCa) reaches
  **50.13%**, +5.2 points over B3 CE (44.9%) and +5.7 over X2 logit-KD (44.4%)
  — the first row to clear §5.1's ≥2-point bar. Single seed (§5). See §12.
* **D3 (CE-only on the aligned vision tower) will not be run** — author
  decision, 2026-09-07, so the paper cannot attribute D5's +5.2 points between
  the vision-alignment stage and the answer-distribution KD stage; see §12 for
  the reasoning and the stated limitation. Nothing else in the ladder needs
  D3, so this is a scope decision, not a blocked measurement.
* **Retraining variability is no longer measured.** §5 moved to a single seed on
  2026-09-07. The one anchor that exists is B3 at 44.9% / 44.0% across seeds
  17/42 — a 0.9-point spread, wider than most differences the ladder is trying to
  resolve. Every sub-point comparison in this project should be read against that
  number, and the manuscript owes it as a stated limitation.
* Test split remains unscored, as §5 requires until G5 locks.
