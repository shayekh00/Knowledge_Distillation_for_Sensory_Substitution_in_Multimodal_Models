# 20260905-none2qw9b-T1-s17-1f44f758

**Teacher zero-shot depth, NF4 (T1)**

| Field | Value |
|---|---|
| Recipe | `T1` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-9B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T21:12:45+00:00 |

**Macro accuracy: 44.7%**  ·  invalid outputs: 3.7%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 59.4% | 0.0% |
| identify_superlative | 341 | 17.6% | 10.9% |
| left_right | 340 | 61.2% | 0.0% |
| nearest_object | 343 | 15.5% | 7.9% |
| relative_depth | 344 | 70.1% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

NF4 quantized, thinking disabled. PILOT and quantization-specific. Section 10.3 teacher suitability.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw9b-T1-s17-1f44f758/predictions.csv \
    --split val --model-name "Teacher zero-shot depth, NF4 (T1)"
```
