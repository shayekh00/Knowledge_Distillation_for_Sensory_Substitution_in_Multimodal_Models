# 20260917-qw9b2qw08b-JKD-s17-b1f44e9b

**J-KD: CE + raw X-Token KD, no feature alignment, vision+language trainable, depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `JKD` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-17T02:55:27+00:00 |

**Macro accuracy: 54.1%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **54.1%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 73.8% | 0.0% |
| identify_superlative | 2490 | 25.9% | 1.0% |
| left_right | 2470 | 75.3% | 0.0% |
| nearest_object | 2474 | 19.9% | 0.8% |
| relative_depth | 2471 | 75.2% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

J grid cell (experiment_protocol.md §8.0/§13 2026-09-16). One confirmatory test pass, predeclared before training and not conditioned on this row's val score.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260917-qw9b2qw08b-JKD-s17-b1f44e9b/predictions.csv \
    --split test --model-name "J-KD: CE + raw X-Token KD, no feature alignment, vision+language trainable, depth, seed 17 -- TEST split"
```
