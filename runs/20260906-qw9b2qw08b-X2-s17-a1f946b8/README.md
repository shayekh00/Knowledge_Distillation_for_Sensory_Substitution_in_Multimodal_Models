# 20260906-qw9b2qw08b-X2-s17-a1f946b8

**X2 CE+X-Token KD, depth, seed 17, lr=2e-5, cached bf16 9B teacher K=4096**

| Field | Value |
|---|---|
| Recipe | `X2` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-06T21:51:05+00:00 |

**Macro accuracy: 42.5%**  ·  invalid outputs: 0.1%

Constrained (closed types snapped to the answer space): **42.5%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 57.4% | 0.0% |
| identify_superlative | 341 | 15.8% | 0.0% |
| left_right | 340 | 57.9% | 0.0% |
| nearest_object | 343 | 12.8% | 0.3% |
| relative_depth | 344 | 68.6% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-qw9b2qw08b-X2-s17-a1f946b8/predictions.csv \
    --split val --model-name "X2 CE+X-Token KD, depth, seed 17, lr=2e-5, cached bf16 9B teacher K=4096"
```
