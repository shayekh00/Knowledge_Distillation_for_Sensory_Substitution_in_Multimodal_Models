# 20260910-qw9b2qw08b-D8-s17-06ef0a0f

**D8 F(cosine)->S2 CE + LoCa X-Token KD, depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `D8` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T19:56:51+00:00 |

**Macro accuracy: 36.3%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **36.3%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 51.0% | 0.3% |
| identify_superlative | 2490 | 8.0% | 0.0% |
| left_right | 2470 | 50.0% | 0.0% |
| nearest_object | 2474 | 11.4% | 1.1% |
| relative_depth | 2471 | 60.9% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5. D2/D8 trained and scored after Part 1's primary-set test score, but their design (recipe, hyperparameters, parent checkpoint) was fixed before any test number existed for either row, so this does not reopen or retune anything already locked.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D8-s17-06ef0a0f/predictions.csv \
    --split test --model-name "D8 F(cosine)->S2 CE + LoCa X-Token KD, depth, seed 17 -- TEST split"
```
