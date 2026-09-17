# 20260916-none2qw08b-J0-s17-d2e993b6

**J0 matched CE control on D7r's surface (CE only, vision+language trainable), depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `J0` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-16T23:35:55+00:00 |

**Macro accuracy: 53.1%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **53.1%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 72.9% | 0.0% |
| identify_superlative | 2490 | 26.1% | 0.3% |
| left_right | 2470 | 73.7% | 0.1% |
| nearest_object | 2474 | 17.7% | 0.8% |
| relative_depth | 2471 | 74.9% | 0.2% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

J grid cell (experiment_protocol.md §8.0/§13 2026-09-16). One confirmatory test pass, predeclared before training and not conditioned on this row's val score.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260916-none2qw08b-J0-s17-d2e993b6/predictions.csv \
    --split test --model-name "J0 matched CE control on D7r's surface (CE only, vision+language trainable), depth, seed 17 -- TEST split"
```
