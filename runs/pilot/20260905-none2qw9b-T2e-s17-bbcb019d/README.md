# 20260905-none2qw9b-T2e-s17-bbcb019d

**Teacher zero-shot RGB, section 6.4 enumerated prompt (T2e)**

| Field | Value |
|---|---|
| Recipe | `T2e` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-9B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `enumerated` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T21:12:49+00:00 |

**Macro accuracy: 58.5%**  ·  invalid outputs: 9.4%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 79.0% | 6.5% |
| identify_superlative | 341 | 28.4% | 19.4% |
| left_right | 340 | 83.2% | 0.0% |
| nearest_object | 343 | 23.3% | 21.0% |
| relative_depth | 344 | 78.5% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

NF4 quantized, thinking disabled. PILOT and quantization-specific. Section 10.3 teacher suitability.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw9b-T2e-s17-bbcb019d/predictions.csv \
    --split val --model-name "Teacher zero-shot RGB, section 6.4 enumerated prompt (T2e)"
```
