# 20260910-qw9b2qw08b-D8-s17-f8da9324

**D8 F(cosine)->S2 CE + LoCa X-Token KD, depth, seed 17, confirmatory**

| Field | Value |
|---|---|
| Recipe | `D8` |
| Status | **CONFIRMATORY** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-10T19:33:05+00:00 |

**Macro accuracy: 36.3%**  ·  invalid outputs: 0.3%

Constrained (closed types snapped to the answer space): **36.3%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 50.3% | 0.9% |
| identify_superlative | 341 | 7.6% | 0.0% |
| left_right | 340 | 47.9% | 0.0% |
| nearest_object | 343 | 11.4% | 0.9% |
| relative_depth | 344 | 64.2% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 33.7%

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260910-qw9b2qw08b-D8-s17-f8da9324/predictions.csv \
    --split val --model-name "D8 F(cosine)->S2 CE + LoCa X-Token KD, depth, seed 17, confirmatory"
```
