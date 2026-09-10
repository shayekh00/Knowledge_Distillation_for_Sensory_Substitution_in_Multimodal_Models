# 20260910-none2qw08b-B3-s17-3da21a87

**B3 depth CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint -- primary comparator, TEST split**

| Field | Value |
|---|---|
| Recipe | `B3` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T01:06:13+00:00 |

**Macro accuracy: 43.6%**  ·  invalid outputs: 0.2%

Constrained (closed types snapped to the answer space): **43.6%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 62.4% | 0.0% |
| identify_superlative | 2490 | 16.2% | 0.1% |
| left_right | 2470 | 61.2% | 0.0% |
| nearest_object | 2474 | 12.4% | 0.8% |
| relative_depth | 2471 | 65.7% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5: 'Once, on the selected checkpoint, after settings are locked. No re-selection after seeing test output.' Settings locked 2026-09-09 (primary recipe D7 chosen on val evidence only, before this score existed).

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-none2qw08b-B3-s17-3da21a87/predictions.csv \
    --split test --model-name "B3 depth CE, seed 17, lr=2e-5, up to 10 epochs patience 2, best-epoch checkpoint -- primary comparator, TEST split"
```
