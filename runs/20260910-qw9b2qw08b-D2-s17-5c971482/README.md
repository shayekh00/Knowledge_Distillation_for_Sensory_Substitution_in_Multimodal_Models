# 20260910-qw9b2qw08b-D2-s17-5c971482

**D2 S2 CE + LoCa X-Token KD, depth, UNALIGNED vision, seed 17, confirmatory**

| Field | Value |
|---|---|
| Recipe | `D2` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T19:05:05+00:00 |

**Macro accuracy: 45.5%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **45.6%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 58.5% | 0.3% |
| identify_superlative | 341 | 20.8% | 0.0% |
| left_right | 340 | 59.7% | 0.0% |
| nearest_object | 343 | 16.6% | 1.2% |
| relative_depth | 344 | 71.8% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D2-s17-5c971482/predictions.csv \
    --split val --model-name "D2 S2 CE + LoCa X-Token KD, depth, UNALIGNED vision, seed 17, confirmatory"
```
