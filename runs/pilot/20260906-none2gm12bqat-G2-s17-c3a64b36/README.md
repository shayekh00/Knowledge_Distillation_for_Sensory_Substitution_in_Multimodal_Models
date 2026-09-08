# 20260906-none2gm12bqat-G2-s17-c3a64b36

**Gemma-4-12B QAT teacher screen, RGB, NF4 (decisive modality for a teacher)**

| Field | Value |
|---|---|
| Recipe | `G2` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `google/gemma-4-12B-it-qat-q4_0-unquantized` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T15:12:33+00:00 |

**Macro accuracy: 39.4%**  ·  invalid outputs: 34.0%

Constrained (closed types snapped to the answer space): **56.3%** — a +16.9% gap, i.e. this model often names the right option without answering in the frozen format

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 40.9% | 50.9% |
| identify_superlative | 341 | 19.4% | 40.8% |
| left_right | 340 | 37.6% | 58.2% |
| nearest_object | 343 | 21.6% | 20.1% |
| relative_depth | 344 | 77.3% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2gm12bqat-G2-s17-c3a64b36/predictions.csv \
    --split val --model-name "Gemma-4-12B QAT teacher screen, RGB, NF4 (decisive modality for a teacher)"
```
