# 20260910-qw9b2qw08b-D7-s17-6a0159b1

**D7 joint feature+CE+LoCa X-Token KD, depth, seed 17 -- primary/reported recipe, TEST split**

| Field | Value |
|---|---|
| Recipe | `D7` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T03:53:43+00:00 |

**Macro accuracy: 54.1%**  ·  invalid outputs: 0.4%

Constrained (closed types snapped to the answer space): **54.1%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 71.7% | 0.0% |
| identify_superlative | 2490 | 29.4% | 0.8% |
| left_right | 2470 | 75.6% | 0.0% |
| nearest_object | 2474 | 19.6% | 1.0% |
| relative_depth | 2471 | 74.2% | 0.1% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5: 'Once, on the selected checkpoint, after settings are locked. No re-selection after seeing test output.' Settings locked 2026-09-09 (primary recipe D7 chosen on val evidence only, before this score existed).

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D7-s17-6a0159b1/predictions.csv \
    --split test --model-name "D7 joint feature+CE+LoCa X-Token KD, depth, seed 17 -- primary/reported recipe, TEST split"
```
