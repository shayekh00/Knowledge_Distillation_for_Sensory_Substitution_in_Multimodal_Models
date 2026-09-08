# 20260906-none2qw08b-B3-s17-0b3dc085

**B3 depth CE LR sweep candidate 2e-5, batch 1 (tuning)**

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
| Recorded (UTC) | 2026-09-06T14:30:37+00:00 |

**Macro accuracy: 42.8%**  ·  invalid outputs: 0.1%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 57.4% | 0.0% |
| identify_superlative | 341 | 16.4% | 0.0% |
| left_right | 340 | 58.5% | 0.0% |
| nearest_object | 343 | 13.4% | 0.0% |
| relative_depth | 344 | 68.3% | 0.3% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B3-s17-0b3dc085/predictions.csv \
    --split val --model-name "B3 depth CE LR sweep candidate 2e-5, batch 1 (tuning)"
```
