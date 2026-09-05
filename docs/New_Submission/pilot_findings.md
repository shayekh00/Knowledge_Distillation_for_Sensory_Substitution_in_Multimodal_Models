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

**The bottleneck is the data path, not the GPU.** Utilisation sat at ~19% during
training: depth decoding, the Prewitt pass, and image preprocessing all run inline
and single-threaded. Moving them to a `DataLoader` with workers should give a
3-5x speedup — plausibly 30 minutes per epoch rather than 100. That materially
changes the cost of the confirmatory core and should be fixed before any large
sweep, since the "weeks of wall-clock" estimate assumes the current throughput.

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

§6.4 requires **one** instruction for every model, so the wording cannot be
chosen per model. `terse` rescues the 0.8B student, but it has not been checked on
the 9B teacher, and a wording that helps a small student could plausibly hurt a
large one. Before the prompt is frozen:

1. Run the same three-way comparison on the teacher.
2. Choose on validation, then freeze — §6.4 forbids retrofitting after test
   results are seen. Doing this now, pre-freeze, on `val`, is the legitimate
   moment.
3. Consider whether constrained decoding (§6.3, `evaluation/candidate_scoring.py`)
   makes the choice less load-bearing. It should: a token trie makes an illegal
   answer unreachable regardless of how the instruction is worded.

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

---

## Open

* B3 (depth CE LoRA fine-tune) is running; B4 follows.
* Teacher prompt comparison, before §6.4 is frozen.
* Teacher suitability on depth (§10.3) — a precondition of the whole study, and
  inference-only, so it needs no 4090.
