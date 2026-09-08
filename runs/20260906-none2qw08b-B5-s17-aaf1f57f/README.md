# 20260906-none2qw08b-B5-s17-aaf1f57f

**B5 RGB CE, seed 17, lr=2e-5, batch 1 (cross-check of batch invariance)**

| Field | Value |
|---|---|
| Recipe | `B5` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T14:30:04+00:00 |

**Macro accuracy: 60.2%**  ·  invalid outputs: 2.1%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 83.2% | 0.0% |
| identify_superlative | 341 | 28.2% | 0.0% |
| left_right | 340 | 84.1% | 2.9% |
| nearest_object | 343 | 26.8% | 7.6% |
| relative_depth | 344 | 78.5% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B5-s17-aaf1f57f/predictions.csv \
    --split val --model-name "B5 RGB CE, seed 17, lr=2e-5, batch 1 (cross-check of batch invariance)"
```
