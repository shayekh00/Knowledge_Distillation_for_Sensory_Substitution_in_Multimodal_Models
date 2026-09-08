# 20260907-qw9b2qw08b-X2-s17-c0612fef

**X2 CE+X-Token KD, depth, seed 17, up to 10 epochs patience 2, best-epoch checkpoint**

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
| Recorded (UTC) | 2026-09-07T03:02:25+00:00 |

**Macro accuracy: 44.4%**  ·  invalid outputs: 0.2%

Constrained (closed types snapped to the answer space): **44.4%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 57.7% | 0.0% |
| identify_superlative | 341 | 17.6% | 0.3% |
| left_right | 340 | 60.6% | 0.0% |
| nearest_object | 343 | 15.5% | 0.6% |
| relative_depth | 344 | 70.6% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260907-qw9b2qw08b-X2-s17-c0612fef/predictions.csv \
    --split val --model-name "X2 CE+X-Token KD, depth, seed 17, up to 10 epochs patience 2, best-epoch checkpoint"
```
