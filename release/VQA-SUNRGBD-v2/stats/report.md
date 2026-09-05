# VQA-SUNRGBD-v2 v2.4 — release statistics

Frozen 2026-09-05T01:10:06.500159+00:00 · license CC BY-SA 4.0 · source: SUN RGB-D (Song et al., CVPR 2015)

## Size

| Split | Items | Images | existence | identify_superlative | left_right | nearest_object | relative_depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 15,278 | 4,187 | 4186 | 2615 | 2995 | 2758 | 2724 |
| val | 1,720 | 667 | 352 | 341 | 340 | 343 | 344 |
| test | 12,463 | 4,703 | 2558 | 2490 | 2470 | 2474 | 2471 |

## Answer balance (test)

| Type | Distinct answers | Majority answer | Majority share |
|---|---:|---|---:|
| existence | 2 | `yes` | 50.0% |
| identify_superlative | 129 | `table` | 8.0% |
| left_right | 2 | `right` | 50.0% |
| nearest_object | 134 | `chair` | 7.8% |
| relative_depth | 130 | `table` | 9.2% |

## Sensor composition

| Split | kv1 | kv2 | realsense | xtion |
|---|---:|---:|---:|---:|
| train | 3545 | 5923 | 1485 | 4325 |
| val | 451 | 706 | 155 | 408 |
| test | 2504 | 4451 | 1256 | 4252 |

## Drop log (§8.2)

Every candidate rejected by a gate, by reason code.

| Reason | Count |
|---|---:|
| `MARGIN_FAIL` | 22,769 |
| `INVALID_POLYGON` | 10,893 |
| `FEWER_THAN_TWO_SINGLE_INSTANCE_OBJECTS` | 2,389 |
| `INSUFFICIENT_SINGLE_INSTANCE_OBJECTS` | 2,262 |
| `INSUFFICIENT_CANDIDATES` | 1,523 |
| `NO_PAIR_CLEARS_DEPTH_GAP` | 935 |
| `NO_SINGLE_INSTANCE_ANCHOR` | 863 |
| `NO_PAIR_CLEARS_GATES` | 477 |
| `NO_ELIGIBLE_OBJECTS` | 315 |
| `SEQUENCE_SHARED_WITH_TEST` | 302 |
| `ANSWER_IN_QUESTION` | 171 |
| `ANNOTATION_UNPARSEABLE` | 40 |

Full table: `stats/drops.csv` (42,939 rows).

## Known limitations

* existence negatives are matched on canonical name only, so a scene annotated `desk`/`counter`/`coffeetable` can carry a gold `no` for `table` (defect D17; ~902 items release-wide, see DATASET_CREATION_PLAN.md §13.22)
* single-reviewer gold verification only; no inter-rater or kappa claim (§8.3)
* some items reference an object largely outside the frame; no filter for them could be validated (defect D16, §13.17)
* re-deriving the vocabulary from scratch yields 148 concepts rather than the shipped 151; use the committed file (§13.18)

Baselines and the evaluation protocol: `evaluate.py` (§9). Gold verification: `audit/results/report.md` (§8.3).
