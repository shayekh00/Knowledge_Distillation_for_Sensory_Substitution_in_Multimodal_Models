# VQA-SUNRGBD-v2 — evaluation on `test`

| Type | n | Random | Majority | Question-only |
|---|---:|---:|---:|---:|
| existence | — | 49.9% | 50.0% | 50.0% |
| identify_superlative | — | 0.0% | 8.0% | 10.2% |
| left_right | — | 50.7% | 50.0% | 50.4% |
| nearest_object | — | 0.0% | 7.8% | 11.9% |
| relative_depth | — | 50.6% | 49.8% | 52.1% |

Macro accuracy (random): 30.2%
Macro accuracy (majority): 33.1%
Macro accuracy (question-only): 34.9%

Macro-F1 is reported for closed answer spaces only; for `relative_depth` the classes are answer *position* (first / second named object), because its answer space is item-specific.
