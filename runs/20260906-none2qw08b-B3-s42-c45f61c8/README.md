# 20260906-none2qw08b-B3-s42-c45f61c8

**B3 depth CE, seed 42, lr=2e-5, batch 1 (cross-check of batch invariance)**

| Field | Value |
|---|---|
| Recipe | `B3` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 42 |
| Recorded (UTC) | 2026-09-06T14:29:43+00:00 |

**Macro accuracy: 42.5%**  ·  invalid outputs: 0.0%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 57.1% | 0.0% |
| identify_superlative | 341 | 15.0% | 0.0% |
| left_right | 340 | 58.2% | 0.0% |
| nearest_object | 343 | 13.7% | 0.0% |
| relative_depth | 344 | 68.6% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B3-s42-c45f61c8/predictions.csv \
    --split val --model-name "B3 depth CE, seed 42, lr=2e-5, batch 1 (cross-check of batch invariance)"
```
