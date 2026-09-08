# 20260906-none2qw08b-B1-s17-a8b43771

**Zero-shot depth reference (B1)**

| Field | Value |
|---|---|
| Recipe | `B1` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T14:28:37+00:00 |

**Macro accuracy: 36.2%**  ·  invalid outputs: 12.4%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 54.3% | 0.0% |
| identify_superlative | 341 | 10.9% | 39.9% |
| left_right | 340 | 55.9% | 0.0% |
| nearest_object | 343 | 6.4% | 11.4% |
| relative_depth | 344 | 53.5% | 11.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B1-s17-a8b43771/predictions.csv \
    --split val --model-name "Zero-shot depth reference (B1)"
```
