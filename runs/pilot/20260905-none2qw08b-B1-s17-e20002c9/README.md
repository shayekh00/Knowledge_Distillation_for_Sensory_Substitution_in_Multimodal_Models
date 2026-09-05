# 20260905-none2qw08b-B1-s17-e20002c9

**Zero-shot depth reference (B1)**

| Field | Value |
|---|---|
| Recipe | `B1` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T17:27:47+00:00 |

**Macro accuracy: 36.4%**  ·  invalid outputs: 12.4%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 54.5% | 0.0% |
| identify_superlative | 341 | 11.4% | 40.2% |
| left_right | 340 | 55.0% | 0.0% |
| nearest_object | 343 | 6.4% | 11.4% |
| relative_depth | 344 | 54.4% | 10.8% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

No target-task training. Establishes the floor that any depth-adapted student must beat; per plan section 14, a method that only beats this may claim adaptation to depth, not that distillation beats supervision.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw08b-B1-s17-e20002c9/predictions.csv \
    --split val --model-name "Zero-shot depth reference (B1)"
```
