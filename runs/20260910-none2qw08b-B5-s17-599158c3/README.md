# 20260910-none2qw08b-B5-s17-599158c3

**B5 RGB CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint -- TEST split**

| Field | Value |
|---|---|
| Recipe | `B5` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `rgb` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T01:30:10+00:00 |

**Macro accuracy: 59.9%**  ·  invalid outputs: 2.4%

Constrained (closed types snapped to the answer space): **59.9%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 74.9% | 0.0% |
| identify_superlative | 2490 | 35.2% | 0.2% |
| left_right | 2470 | 85.6% | 0.2% |
| nearest_object | 2474 | 24.7% | 11.4% |
| relative_depth | 2471 | 79.0% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5: 'Once, on the selected checkpoint, after settings are locked. No re-selection after seeing test output.' Settings locked 2026-09-09 (primary recipe D7 chosen on val evidence only, before this score existed).

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-none2qw08b-B5-s17-599158c3/predictions.csv \
    --split test --model-name "B5 RGB CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint -- TEST split"
```
