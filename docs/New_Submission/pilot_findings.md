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

## 5. Environment findings

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
