# 20260911-qw9b2qw08b-D7r-s17-059f7e5f

**D7r joint feature+CE+raw KD (D7's LoCa-vs-raw twin, use_loca=False), depth, seed 17 -- TEST split, one-shot per section 13's predeclared decision rule**

| Field | Value |
|---|---|
| Recipe | `D7r` |
| Status | **CONFIRMATORY** |
| Split | `test` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `Qwen/Qwen3.5-9B` |
| Inference modality | `depth` |
| Distillation mode | `xtoken` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-11T05:03:35+00:00 |

**Macro accuracy: 55.4%**  ·  invalid outputs: 0.5%

Constrained (closed types snapped to the answer space): **55.4%** — unchanged, so the model already complies

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 2558 | 74.4% | 0.0% |
| identify_superlative | 2490 | 28.5% | 1.2% |
| left_right | 2470 | 77.4% | 0.0% |
| nearest_object | 2474 | 20.3% | 1.2% |
| relative_depth | 2471 | 76.3% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 30.4%, majority 33.1%, question_only 34.9%

## Notes

One-shot test-split score. Section 13 (2026-09-10 D7r row): D7r's val macro (56.8%) was within noise of D7's (56.82%, paired bootstrap diff +0.41pp, 95% CI [-1.15,+1.97], includes zero) -- so D7r replaces D7 as the reported recipe on parsimony grounds and receives exactly this one confirmatory test pass. D7's own test score (54.1%) was not used to make that call and is not revisited here.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260911-qw9b2qw08b-D7r-s17-059f7e5f/predictions.csv \
    --split test --model-name "D7r joint feature+CE+raw KD (D7's LoCa-vs-raw twin, use_loca=False), depth, seed 17 -- TEST split, one-shot per section 13's predeclared decision rule"
```
