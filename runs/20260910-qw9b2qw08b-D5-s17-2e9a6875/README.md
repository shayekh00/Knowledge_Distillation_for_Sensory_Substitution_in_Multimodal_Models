# 20260910-qw9b2qw08b-D5-s17-2e9a6875

**D5 F->S2 CE + LoCa X-Token KD, depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `D5` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T02:42:07+00:00 |

**Macro accuracy: 50.3%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **50.3%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 71.0% | 0.1% |
| identify_superlative | 2490 | 24.0% | 0.3% |
| left_right | 2470 | 69.4% | 0.0% |
| nearest_object | 2474 | 18.0% | 1.1% |
| relative_depth | 2471 | 69.3% | 0.1% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5: 'Once, on the selected checkpoint, after settings are locked. No re-selection after seeing test output.' Settings locked 2026-09-09 (primary recipe D7 chosen on val evidence only, before this score existed).

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D5-s17-2e9a6875/predictions.csv \
    --split test --model-name "D5 F->S2 CE + LoCa X-Token KD, depth, seed 17 -- TEST split"
```
