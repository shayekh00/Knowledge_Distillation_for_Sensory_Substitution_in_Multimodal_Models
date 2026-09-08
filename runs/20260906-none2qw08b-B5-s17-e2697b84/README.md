# 20260906-none2qw08b-B5-s17-e2697b84

**B5 RGB CE matched student, seed 17, lr=2e-5, batch 4, enable_thinking fix**

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
| Recorded (UTC) | 2026-09-06T18:11:39+00:00 |

**Macro accuracy: 59.8%**  ·  invalid outputs: 2.7%

Constrained (closed types snapped to the answer space): **60.5%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 83.8% | 0.0% |
| identify_superlative | 341 | 28.7% | 0.3% |
| left_right | 340 | 83.2% | 3.8% |
| nearest_object | 343 | 26.2% | 9.6% |
| relative_depth | 344 | 77.0% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260906-none2qw08b-B5-s17-e2697b84/predictions.csv \
    --split val --model-name "B5 RGB CE matched student, seed 17, lr=2e-5, batch 4, enable_thinking fix"
```
