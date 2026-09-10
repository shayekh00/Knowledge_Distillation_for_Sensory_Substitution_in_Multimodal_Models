# 20260910-qw9b2qw08b-D9-s17-79e7b076

**D9 F->S2 teacher-prefix raw X-Token KD only, no gold access, depth, seed 17 -- TEST split**

| Field | Value |
|---|---|
| Recipe | `D9` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T03:29:47+00:00 |

**Macro accuracy: 48.4%**  ·  invalid outputs: 0.9%

Constrained (closed types snapped to the answer space): **48.4%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 72.7% | 0.0% |
| identify_superlative | 2490 | 20.5% | 1.4% |
| left_right | 2470 | 68.3% | 0.0% |
| nearest_object | 2474 | 14.1% | 2.9% |
| relative_depth | 2471 | 66.5% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score, experiment_protocol.md section 5: 'Once, on the selected checkpoint, after settings are locked. No re-selection after seeing test output.' Settings locked 2026-09-09 (primary recipe D7 chosen on val evidence only, before this score existed).

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D9-s17-79e7b076/predictions.csv \
    --split test --model-name "D9 F->S2 teacher-prefix raw X-Token KD only, no gold access, depth, seed 17 -- TEST split"
```
