# 20260907-none2qw08b-B3-s42-594b204b

**B3 depth CE, seed 42, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint**

| Field | Value |
|---|---|
| Recipe | `B3` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 42 |
| Recorded (UTC) | 2026-09-07T08:51:52+00:00 |

**Macro accuracy: 44.0%**  ·  invalid outputs: 0.1%

Constrained (closed types snapped to the answer space): **44.0%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 57.7% | 0.0% |
| identify_superlative | 341 | 17.0% | 0.0% |
| left_right | 340 | 60.6% | 0.0% |
| nearest_object | 343 | 14.0% | 0.6% |
| relative_depth | 344 | 70.6% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260907-none2qw08b-B3-s42-594b204b/predictions.csv \
    --split val --model-name "B3 depth CE, seed 42, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint"
```
