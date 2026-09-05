# 20260905-none2qw08b-B1e-s17-8355a1d5

**Zero-shot depth, section 6.4 enumerated prompt (B1e)**

| Field | Value |
|---|---|
| Recipe | `B1e` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `enumerated` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T17:30:02+00:00 |

**Macro accuracy: 23.0%**  ·  invalid outputs: 59.7%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 56.8% | 1.4% |
| identify_superlative | 341 | 0.0% | 100.0% |
| left_right | 340 | 56.5% | 0.0% |
| nearest_object | 343 | 0.0% | 100.0% |
| relative_depth | 344 | 1.7% | 98.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

Prompt-design evidence, not a capability measurement. Identical model, inputs and decoding to B1; only the instruction wording differs. The section 6.4 wording enumerates the legal answers and this 0.8B model echoes that list back. Read the invalid rate, not the accuracy.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw08b-B1e-s17-8355a1d5/predictions.csv \
    --split val --model-name "Zero-shot depth, section 6.4 enumerated prompt (B1e)"
```
