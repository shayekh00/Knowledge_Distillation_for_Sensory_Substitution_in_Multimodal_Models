# 20260905-none2qw08b-B2-s17-3d9bd6ab

**Zero-shot RGB reference (B2)**

| Field | Value |
|---|---|
| Recipe | `B2` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T17:28:03+00:00 |

**Macro accuracy: 43.0%**  ·  invalid outputs: 15.3%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 73.0% | 0.0% |
| identify_superlative | 341 | 22.0% | 24.6% |
| left_right | 340 | 47.6% | 24.4% |
| nearest_object | 343 | 14.6% | 12.8% |
| relative_depth | 344 | 57.6% | 15.1% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

No target-task training. Clean-RGB capability reference; contrasts with B1 to show how much the same model loses when its visual input becomes depth.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw08b-B2-s17-3d9bd6ab/predictions.csv \
    --split val --model-name "Zero-shot RGB reference (B2)"
```
