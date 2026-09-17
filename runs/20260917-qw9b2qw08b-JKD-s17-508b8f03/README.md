# 20260917-qw9b2qw08b-JKD-s17-508b8f03

**J-KD: CE + raw X-Token KD, no feature alignment, vision+language trainable, depth, seed 17 -- VAL split**

| Field | Value |
|---|---|
| Recipe | `JKD` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-17T02:31:33+00:00 |

**Macro accuracy: 54.6%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **54.6%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 70.5% | 0.0% |
| identify_superlative | 341 | 28.2% | 0.6% |
| left_right | 340 | 76.8% | 0.0% |
| nearest_object | 343 | 19.5% | 1.2% |
| relative_depth | 344 | 77.9% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Notes

J grid cell (experiment_protocol.md §8.0): D7r's exact trainable surface, stage, seed, data order, caches and schedule; only the declared loss terms differ.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260917-qw9b2qw08b-JKD-s17-508b8f03/predictions.csv \
    --split val --model-name "J-KD: CE + raw X-Token KD, no feature alignment, vision+language trainable, depth, seed 17 -- VAL split"
```
