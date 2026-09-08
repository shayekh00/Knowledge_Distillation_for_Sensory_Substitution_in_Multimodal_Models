# 20260906-none2qw9b-T2-s17-9600c136

**Teacher zero-shot RGB, bf16 (T2, confirmatory precision)**

| Field | Value |
|---|---|
| Recipe | `T2` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-9B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T14:21:26+00:00 |

**Macro accuracy: 63.1%**  ·  invalid outputs: 4.4%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 85.5% | 0.0% |
| identify_superlative | 341 | 31.7% | 10.3% |
| left_right | 340 | 89.1% | 0.0% |
| nearest_object | 343 | 26.2% | 11.7% |
| relative_depth | 344 | 82.8% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw9b-T2-s17-9600c136/predictions.csv \
    --split val --model-name "Teacher zero-shot RGB, bf16 (T2, confirmatory precision)"
```
