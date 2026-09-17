# 20260917-qw9b2qw08b-JFEAT-s17-ade7f3cd

**J-Feature: CE + contrastive feature alignment, no output KD, vision+language trainable, depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `JFEAT` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-17T05:58:06+00:00 |

**Macro accuracy: 53.2%**  ·  invalid outputs: 1.8%

Constrained (closed types snapped to the answer space): **53.3%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 70.7% | 0.0% |
| identify_superlative | 2490 | 28.3% | 0.4% |
| left_right | 2470 | 74.0% | 0.2% |
| nearest_object | 2474 | 18.3% | 8.6% |
| relative_depth | 2471 | 74.9% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

J grid cell (experiment_protocol.md §8.0/§13 2026-09-16). One confirmatory test pass, predeclared before training and not conditioned on this row's val score.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260917-qw9b2qw08b-JFEAT-s17-ade7f3cd/predictions.csv \
    --split test --model-name "J-Feature: CE + contrastive feature alignment, no output KD, vision+language trainable, depth, seed 17 -- TEST split"
```
