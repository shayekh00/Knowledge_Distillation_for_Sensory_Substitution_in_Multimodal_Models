# 20260906-none2qw08b-B3-s17-a4c0db66

**B3 depth CE LR sweep candidate 5e-6, batch 1 (tuning)**

| Field | Value |
|---|---|
| Recipe | `B3` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T14:30:15+00:00 |

**Macro accuracy: 40.7%**  ·  invalid outputs: 0.2%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 56.5% | 0.0% |
| identify_superlative | 341 | 18.2% | 0.3% |
| left_right | 340 | 57.1% | 0.0% |
| nearest_object | 343 | 13.7% | 0.6% |
| relative_depth | 344 | 58.1% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B3-s17-a4c0db66/predictions.csv \
    --split val --model-name "B3 depth CE LR sweep candidate 5e-6, batch 1 (tuning)"
```
