# 20260907-none2qw08b-B3-s17-e370ce8a

**B3 depth CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint**

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
| Seed | 17 |
| Recorded (UTC) | 2026-09-07T00:28:37+00:00 |

**Macro accuracy: 44.9%**  ·  invalid outputs: 0.0%

Constrained (closed types snapped to the answer space): **44.9%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 58.8% | 0.0% |
| identify_superlative | 341 | 18.2% | 0.0% |
| left_right | 340 | 61.8% | 0.0% |
| nearest_object | 343 | 15.7% | 0.0% |
| relative_depth | 344 | 69.8% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260907-none2qw08b-B3-s17-e370ce8a/predictions.csv \
    --split val --model-name "B3 depth CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint"
```
