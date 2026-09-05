# 20260905-none2qw9b-T2-s17-52b6f906

**Teacher zero-shot RGB, NF4 (T2)**

| Field | Value |
|---|---|
| Recipe | `T2` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-9B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T21:12:41+00:00 |

**Macro accuracy: 62.2%**  ·  invalid outputs: 4.5%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 84.9% | 0.0% |
| identify_superlative | 341 | 29.3% | 12.0% |
| left_right | 340 | 89.7% | 0.0% |
| nearest_object | 343 | 28.3% | 10.5% |
| relative_depth | 344 | 78.8% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

NF4 quantized, thinking disabled. PILOT and quantization-specific. Section 10.3 teacher suitability.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw9b-T2-s17-52b6f906/predictions.csv \
    --split val --model-name "Teacher zero-shot RGB, NF4 (T2)"
```
