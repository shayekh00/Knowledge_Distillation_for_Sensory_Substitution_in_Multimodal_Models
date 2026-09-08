# 20260907-none2qw08b-B5-s17-47e02505

**B5 RGB CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint**

| Field | Value |
|---|---|
| Recipe | `B5` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-07T06:43:26+00:00 |

**Macro accuracy: 62.2%**  ·  invalid outputs: 2.2%

Constrained (closed types snapped to the answer space): **62.2%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 83.0% | 0.0% |
| identify_superlative | 341 | 32.6% | 0.0% |
| left_right | 340 | 86.8% | 0.0% |
| nearest_object | 343 | 27.1% | 10.5% |
| relative_depth | 344 | 81.4% | 0.3% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260907-none2qw08b-B5-s17-47e02505/predictions.csv \
    --split val --model-name "B5 RGB CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint"
```
