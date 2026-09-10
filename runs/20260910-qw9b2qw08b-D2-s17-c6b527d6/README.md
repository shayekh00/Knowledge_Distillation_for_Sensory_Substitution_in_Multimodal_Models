# 20260910-qw9b2qw08b-D2-s17-c6b527d6

**D2 S2 CE + LoCa X-Token KD, depth, UNALIGNED vision, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `D2` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T19:29:11+00:00 |

**Macro accuracy: 44.7%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **44.7%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 62.5% | 0.0% |
| identify_superlative | 2490 | 18.4% | 0.0% |
| left_right | 2470 | 62.0% | 0.0% |
| nearest_object | 2474 | 13.9% | 1.5% |
| relative_depth | 2471 | 66.8% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5. D2/D8 trained and scored after Part 1's primary-set test score, but their design (recipe, hyperparameters, parent checkpoint) was fixed before any test number existed for either row, so this does not reopen or retune anything already locked.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D2-s17-c6b527d6/predictions.csv \
    --split test --model-name "D2 S2 CE + LoCa X-Token KD, depth, UNALIGNED vision, seed 17 -- TEST split"
```
