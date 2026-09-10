# 20260910-qw9b2qw08b-D4-s17-895370c5

**D4 F->S2 CE + raw X-Token KD, depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `D4` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T02:18:11+00:00 |

**Macro accuracy: 50.6%**  ·  invalid outputs: 0.2%

Constrained (closed types snapped to the answer space): **50.6%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 70.4% | 0.0% |
| identify_superlative | 2490 | 23.3% | 0.4% |
| left_right | 2470 | 69.9% | 0.0% |
| nearest_object | 2474 | 17.9% | 0.6% |
| relative_depth | 2471 | 71.7% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5: 'Once, on the selected checkpoint, after settings are locked. No re-selection after seeing test output.' Settings locked 2026-09-09 (primary recipe D7 chosen on val evidence only, before this score existed).

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D4-s17-895370c5/predictions.csv \
    --split test --model-name "D4 F->S2 CE + raw X-Token KD, depth, seed 17 -- TEST split"
```
