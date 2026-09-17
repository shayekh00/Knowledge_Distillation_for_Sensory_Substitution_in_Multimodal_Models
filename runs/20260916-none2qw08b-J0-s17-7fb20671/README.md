# 20260916-none2qw08b-J0-s17-7fb20671

**J0 matched CE control on D7r's surface (CE only, vision+language trainable), depth, seed 17 -- VAL split**

| Field | Value |
|---|---|
| Recipe | `J0` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-16T23:12:06+00:00 |

**Macro accuracy: 54.4%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **54.5%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 70.5% | 0.0% |
| identify_superlative | 341 | 27.6% | 0.0% |
| left_right | 340 | 75.6% | 0.3% |
| nearest_object | 343 | 20.1% | 1.2% |
| relative_depth | 344 | 78.5% | 0.3% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Notes

J grid cell (experiment_protocol.md §8.0): D7r's exact trainable surface, stage, seed, data order, caches and schedule; only the declared loss terms differ.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260916-none2qw08b-J0-s17-7fb20671/predictions.csv \
    --split val --model-name "J0 matched CE control on D7r's surface (CE only, vision+language trainable), depth, seed 17 -- VAL split"
```
