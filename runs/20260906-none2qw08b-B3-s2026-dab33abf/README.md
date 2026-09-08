# 20260906-none2qw08b-B3-s2026-dab33abf

**B3 depth CE, seed 2026, lr=2e-5, batch 4, enable_thinking fix**

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
| Seed | 2026 |
| Recorded (UTC) | 2026-09-06T17:47:04+00:00 |

**Macro accuracy: 42.1%**  ·  invalid outputs: 0.0%

Constrained (closed types snapped to the answer space): **42.1%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 56.8% | 0.0% |
| identify_superlative | 341 | 20.8% | 0.0% |
| left_right | 340 | 57.4% | 0.0% |
| nearest_object | 343 | 13.7% | 0.0% |
| relative_depth | 344 | 61.9% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B3-s2026-dab33abf/predictions.csv \
    --split val --model-name "B3 depth CE, seed 2026, lr=2e-5, batch 4, enable_thinking fix"
```
