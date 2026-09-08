# 20260906-none2qw9b-T1-s17-9504469f

**Teacher zero-shot depth, bf16 (T1, confirmatory precision)**

| Field | Value |
|---|---|
| Recipe | `T1` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-9B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T14:21:15+00:00 |

**Macro accuracy: 43.9%**  ·  invalid outputs: 3.4%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 55.7% | 0.0% |
| identify_superlative | 341 | 17.6% | 9.4% |
| left_right | 340 | 63.8% | 0.0% |
| nearest_object | 343 | 13.7% | 7.9% |
| relative_depth | 344 | 68.9% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw9b-T1-s17-9504469f/predictions.csv \
    --split val --model-name "Teacher zero-shot depth, bf16 (T1, confirmatory precision)"
```
