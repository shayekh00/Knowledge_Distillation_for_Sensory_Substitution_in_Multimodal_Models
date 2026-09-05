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

A full training epoch over the 15,278 train rows projects to **~2 hours**. The
card has roughly 4x headroom over what LoRA training actually uses, and inference
and training were observed running *concurrently* without exhausting it.

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

## 4. Environment findings

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
