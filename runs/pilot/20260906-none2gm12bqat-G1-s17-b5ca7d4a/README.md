# 20260906-none2gm12bqat-G1-s17-b5ca7d4a

**Gemma-4-12B QAT screen, depth (premise check only; KD gives the teacher RGB)**

| Field | Value |
|---|---|
| Recipe | `G1` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `google/gemma-4-12B-it-qat-q4_0-unquantized` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T15:12:53+00:00 |

**Macro accuracy: 32.8%**  ·  invalid outputs: 23.3%

Constrained (closed types snapped to the answer space): **40.0%** — a +7.1% gap, i.e. this model often names the right option without answering in the frozen format

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 47.4% | 10.8% |
| identify_superlative | 341 | 15.0% | 22.3% |
| left_right | 340 | 28.5% | 51.2% |
| nearest_object | 343 | 8.5% | 31.2% |
| relative_depth | 344 | 64.8% | 1.7% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2gm12bqat-G1-s17-b5ca7d4a/predictions.csv \
    --split val --model-name "Gemma-4-12B QAT screen, depth (premise check only; KD gives the teacher RGB)"
```
