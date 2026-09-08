# 20260906-none2qw08b-B3-s42-24795ff4

**B3 depth CE, seed 42, lr=2e-5, batch 4, enable_thinking fix**

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
| Recorded (UTC) | 2026-09-06T17:22:05+00:00 |

**Macro accuracy: 42.0%**  ·  invalid outputs: 0.0%

Constrained (closed types snapped to the answer space): **42.0%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 56.5% | 0.0% |
| identify_superlative | 341 | 15.8% | 0.0% |
| left_right | 340 | 56.2% | 0.0% |
| nearest_object | 343 | 13.1% | 0.0% |
| relative_depth | 344 | 68.3% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B3-s42-24795ff4/predictions.csv \
    --split val --model-name "B3 depth CE, seed 42, lr=2e-5, batch 4, enable_thinking fix"
```
