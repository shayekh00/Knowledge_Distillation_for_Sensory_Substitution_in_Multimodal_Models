# 20260917-qw9b2qw08b-JFEAT-s17-8db9cca0

**J-Feature: CE + contrastive feature alignment, no output KD, vision+language trainable, depth, seed 17 -- VAL split**

| Field | Value |
|---|---|
| Recipe | `JFEAT` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-17T05:34:12+00:00 |

**Macro accuracy: 54.9%**  ·  invalid outputs: 2.3%

Constrained (closed types snapped to the answer space): **54.9%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 72.4% | 0.0% |
| identify_superlative | 341 | 27.0% | 0.0% |
| left_right | 340 | 76.2% | 0.0% |
| nearest_object | 343 | 19.8% | 11.7% |
| relative_depth | 344 | 79.1% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Notes

J grid cell (experiment_protocol.md §8.0): D7r's exact trainable surface, stage, seed, data order, caches and schedule; only the declared loss terms differ.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260917-qw9b2qw08b-JFEAT-s17-8db9cca0/predictions.csv \
    --split val --model-name "J-Feature: CE + contrastive feature alignment, no output KD, vision+language trainable, depth, seed 17 -- VAL split"
```
