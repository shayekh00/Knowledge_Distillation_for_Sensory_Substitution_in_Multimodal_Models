# 20260906-none2qw08b-B2-s17-56c2d083

**Zero-shot RGB reference (B2)**

| Field | Value |
|---|---|
| Recipe | `B2` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T14:28:49+00:00 |

**Macro accuracy: 42.6%**  ·  invalid outputs: 14.8%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 72.4% | 0.0% |
| identify_superlative | 341 | 21.7% | 23.8% |
| left_right | 340 | 48.5% | 23.8% |
| nearest_object | 343 | 14.6% | 12.0% |
| relative_depth | 344 | 55.8% | 15.1% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B2-s17-56c2d083/predictions.csv \
    --split val --model-name "Zero-shot RGB reference (B2)"
```
