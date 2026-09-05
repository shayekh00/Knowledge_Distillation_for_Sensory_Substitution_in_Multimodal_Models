# Dataset protocol — VQA-SUNRGBD v2.4

**Status:** Phase 2 deliverable of `docs/NEW_SUBMISSION.md` §19 · Gate **G1**
**Created:** 2026-09-05
**Companion to:** [`experiment_protocol.md`](experiment_protocol.md)

This is the source for the manuscript's "Task and benchmark" section. Every rule
below is traced to the generator that implements it, so each construction claim
in the paper is checkable against the code rather than against recollection.

> **Scope rule — audit artifacts are excluded.** Per the locked dataset decision
> in `NEW_SUBMISSION.md` §12, nothing derived from `audit/` appears here: no gold
> accuracy rate, no inter-rater figure, no human-verification claim. Some numbers
> of that kind *do* appear in the shipped `DATASHEET.md`; they are deliberately
> not carried into this document, the paper, or the supplement. Limitations below
> are stated structurally — derived from what the generation rule can and cannot
> decide — never from a measured error rate.

---

## 1. Release identity

| Item | Value |
|---|---|
| Version | v2.4, frozen, used unchanged |
| Manifest | `release/VQA-SUNRGBD-v2/FROZEN_v2.4.json` (sha256 `659d65f8…c636f8d2`) |
| Frozen at | 2026-09-05T01:10:06Z |
| Verification | `freeze_release.py --verify v2.4` → "verified, no drift" |
| Source corpus | SUN RGB-D, 10,335 frames; 9,993 scenes survive indexing |
| Splits | train 15,278 · val 1,720 · test 12,463 |

**Description to use in the paper:** *a rule-based VQA benchmark derived from SUN
RGB-D annotations and decoded depth measurements.* Deterministic generation,
fixed release hashes, explicit question definitions, and recorded split grouping
may all be claimed. Human validation may not.

---

## 2. Shared substrate

**Configuration** (`data/config.yaml`): `seed: 42` · `split.val_fraction: 0.15` ·
`depth.clip_max_m: 8.0` · `depth.min_valid_fraction: 0.30` ·
`geometry.min_area_frac: 0.005`.

**Depth decoding** (`build_index.py:114-118`, mirrored in `depth_utils.py:18-22`)
uses the official SUN RGB-D 16-bit rotation `(raw >> 3) | (raw << 13)`, divided by
1000, clipped to `[0, 8.0]` m. Per-object depth is the median over mask pixels
with depth > 0, and is `None` unless at least **30%** of the mask has valid depth.

**Object eligibility** (`scene_objects.py:71-131`), identical for every generator:

```
reference_eligible = valid_polygon AND in_vocab AND area_frac >= 0.005
eligible           = reference_eligible AND NOT is_structural
```

Ten concepts are structural and never answers: `baseboard, ceiling, column,
doorframe, doorway, floor, pillar, stair, tile, wall`. `door` and `window` are
**not** structural and are legal answers.

**Determinism** (`generator_common.py:134`): each scene draws from
`Random(f"{42 + offset}:{question_type}:{image_id}")`, so dropping one scene never
changes the wording of another. Six templates per type/variant, asserted at
`generator_common.py:51`; the chosen index is recorded as `template_id`.

---

## 3. The five question types

### 3.1 `existence` — 4,186 train / 352 val / 2,558 test

| | |
|---|---|
| **Rule** | One *positive* (`yes`) per eligible concept in the scene; one *negative* (`no`) per concept in a plausible-absent pool (`existence.py:121-167`) |
| **Negative pool** | `(scene_type_frequent ∪ category_matched) − present_concepts`, minus structural concepts. `scene_type_frequent` = concept appears in ≥ **5%** of scenes of this `scene_type` in the frozen co-occurrence table; `category_matched` = non-structural concepts sharing a category with an eligible object present |
| **Evidence used** | 2-D polygons and canonical names only, plus `scene_type`. **No depth** |
| **Answer space** | `yes\|no` |
| **Rejection** | `NO_ELIGIBLE_OBJECTS` (scene emits nothing); `NO_HARD_NEGATIVE` (positives kept) |
| **`evidence` field** | positive `{concept, object_index, area_frac}`; negative `{concept, reason:"category_or_scene_type_plausible"}` |

**Worked example** — `test_000004`, realsense, `office`: *"Is there any table visible
in the image?"* → `no`. No annotated object canonicalizes to `table`; `table`
enters the negative pool via the scene-type route (it appears in ~36% of `office`
scenes, above the 5% threshold).

### 3.2 `identify_superlative` — 2,615 / 341 / 2,490

| | |
|---|---|
| **Rule** | Two variants. `closest_camera`: argmin `depth_median_m`, requiring runner-up ≥ **1.2×** winner **and** winner ≥ **0.4 m**. `farthest_camera`: argmax, requiring winner ≥ **1.2×** runner-up (`identify_superlative.py:31-52`) |
| **Constants** | `DEPTH_MARGIN = 1.2`, `MIN_CLOSEST_DEPTH_M = 0.4` — hardcoded in the generator, **not** in `config.yaml` |
| **Evidence used** | Decoded metric depth (per-object median); polygons only gate the pool |
| **Answer space** | Open vocabulary; `answer_space` is **empty** (NaN when read). 129 distinct answers in test |
| **Rejection** | `NO_ELIGIBLE_OBJECTS`, `MARGIN_FAIL`, `ANSWER_IN_QUESTION` |
| **`evidence` field** | `{winner_concept, winner_object_index, winner_area_frac, winner_depth_m, runner_up_concept, runner_up_area_frac, runner_up_depth_m}` |

**Worked example** — `test_000002`, kv2, `rest_space`, `closest_camera` → `table`.
Depth-bearing eligible objects: table 1.520 m, curtain 2.157 m, curtain 2.603 m,
window 2.964 m. Gates: 1.520 ≥ 0.4 ✓; 2.157 ≥ 1.2 × 1.520 = 1.824 ✓.

### 3.3 `left_right` — 2,995 / 340 / 2,470

| | |
|---|---|
| **Rule** | Answer object A from `eligible`; reference B from `reference_eligible` minus `{wall, floor, ceiling}`. Both must be **single-instance**. Answer is `left` iff `centroid_x(A) < centroid_x(B)` (`left_right.py:88-146`) |
| **Gates** | horizontal gap ≥ **10%** of image width; polygon **IoU ≤ 0.20** |
| **Evidence used** | Purely 2-D: polygon area-centroids and image width. Polygons are re-parsed from the raw annotation for the IoU test. **No depth** |
| **Answer space** | `left\|right` |
| **Rejection** | `INSUFFICIENT_SINGLE_INSTANCE_OBJECTS`, `POLYGON_UNAVAILABLE`, `NO_PAIR_CLEARS_GATES` |
| **`evidence` field** | `{a_concept, b_concept, a_centroid_x, b_centroid_x, horizontal_gap_px, image_width}` — the IoU value is **not** recorded |

**Worked example** — `test_000000`, kv2, `dining_room`: *"Is the frame to the left or
to the right of the carpet?"* → `right`. `centroid_x` 418.60 vs 102.48, gap 316.12 ≥
0.10 × 730 = 73.0 ✓, IoU 0.0 ✓, and 418.60 > 102.48 ⇒ `right`.

**Note for the paper:** `left_right` is solvable from 2-D layout alone and is *not*
evidence of metric-depth reasoning. It must be reported separately from the
depth-relation aggregate (`experiment_protocol.md` §2.1).

### 3.4 `nearest_object` — 2,758 / 343 / 2,474

| | |
|---|---|
| **Rule** | For each single-instance anchor, rank other objects by Euclidean distance in the camera frame; the winner must satisfy `nearest ≤ 0.8 × second_nearest` (`nearest_object.py:32-86`). Candidates of the anchor's own concept are excluded |
| **Evidence used** | Polygon area-centroid + `depth_median_m`, back-projected through the scene's `intrinsics.txt` with a pinhole model. `Rtilt` is deliberately skipped — a shared rotation preserves pairwise distance |
| **Answer space** | Open vocabulary; `answer_space` **empty**. 134 distinct answers in test |
| **Rejection** | `NO_SINGLE_INSTANCE_ANCHOR`, `MISSING_INTRINSICS`, `INSUFFICIENT_CANDIDATES`, `MARGIN_FAIL`, `ANSWER_IN_QUESTION` |
| **`evidence` field** | `{anchor_concept, anchor_object_index, answer_concept, answer_object_index, nearest_distance_m, second_nearest_distance_m}` |

**Worked example** — `test_000005`, kv1 `NYUdata/NYU0056`, `bedroom`: *"What is the
closest object to the night stand?"* → `tissue box`. Back-projection with
`fx=518.86, fy=519.47` gives anchor `(-0.889, 0.510, 1.839)` and tissue box
`(-1.025, 0.151, 2.049)`; distance 0.438 m ≤ 0.8 × 0.662 = 0.530 ✓.

### 3.5 `relative_depth` — 2,724 / 344 / 2,471

| | |
|---|---|
| **Rule** | All pairs of single-instance eligible objects with depth, requiring `\|d(A) − d(B)\| ≥ max(0.3 m, 0.15 × min(d(A), d(B)))` (`relative_depth.py:27-29`) |
| **Randomisation** | Two independent coin flips per surviving pair: mention order, and polarity (`closer` / `farther`) |
| **Evidence used** | `depth_median_m` for both objects only |
| **Answer space** | **Item-specific**: the two objects' display names joined in *question mention order*, e.g. `chair\|table` |
| **Rejection** | `FEWER_THAN_TWO_SINGLE_INSTANCE_OBJECTS`, `NO_PAIR_CLEARS_DEPTH_GAP` |
| **`evidence` field** | `{a_concept, a_depth_m, b_concept, b_depth_m, comparative, answer_concept}` |

**Worked example** — `test_000001`, kv2, `classroom`: *"Which one is closer, the chair
or the table?"* → `table`, `answer_space = chair|table`. Depths 1.900 vs 1.019;
gap 0.881 ≥ max(0.3, 0.153) ✓; `closer` ⇒ argmin ⇒ `table`.

**Trap — evidence order is not mention order.** `evidence.a_concept`/`b_concept`
record the pair-enumeration order, which disagrees with the question's mention
order in roughly half of all rows. Anything reconstructing mention order (the
first/second-mentioned label used for macro-F1) **must read `answer_space`**, not
the evidence field.

---

## 4. Balancing and stratification

`balance.py` only ever *drops* rows — nothing is invented or duplicated.

**Per type** (`build_release.py:61-108`):

- **existence** — greedy best-of-64 pairing of one `yes` with one `no` per concept,
  with each image used at most once across the whole frame. Applied in all splits.
  Verified: every concept has exactly `yes == no` in test.
- **identify_superlative** — one row per image; uniform answer target in val/test,
  then a majority-share cap of **8%**.
- **nearest_object** — one row per image with uniform target in all splits; a **20%**
  answer cap conditioned on anchor concept in all splits; 8% majority cap in val/test.
- **left_right** — one row per image, then binary balancing in val/test, applied a
  second time after sensor subsampling because subsampling reintroduces drift.
- **relative_depth** — one row per image and **no answer balancing**: the first/second
  mention label is ~50/50 by construction (1,231/1,240 in test).

**Joint val/test stage** (`build_release.py:127-155`): all five types are subsampled
to the smallest type's size, stratified by **sensor**; then at most **4 distinct
question types per image** are kept, with `existence` protected from removal. This
protection is why existence has 2,558 test rows while the other four have
2,470–2,490.

Train receives none of the joint stage: no sensor subsample, no per-image type cap,
no answer targeting.

`question_id` is assigned last, as `f"{split}_{i:06d}"` over the shuffled frame — so
**any rebuild reassigns every id.**

**Release-blocking sanity checks** (`build_release.py:176-313`) abort the build on:
duplicate `(image_id, question)`; a missing RGB or depth file; an answer outside a
non-empty `answer_space`; a surviving retired `largest` variant; existence image
reuse or `yes != no`; and **any `image_id` or `sequence_id` shared between two
splits**. v2.4 reports zero failures and zero warnings.

---

## 5. Split procedure

1. `sequence_id` = `sun3d:<building>/<room>` for SUN3D paths, else `scene:<image_id>`.
2. `test` is the **official** SUN RGB-D test split from `allsplit.mat`, never trimmed.
3. Any official-train image whose `sequence_id` also contains a test frame is
   dropped entirely (302 images, `SEQUENCE_SHARED_WITH_TEST`).
4. The remainder is split with `GroupShuffleSplit(test_size=0.15, random_state=42)`
   grouped on `sequence_id`.
5. Disjointness is re-verified downstream as a hard build failure.

**Claim boundary.** This verifies the *recorded* grouping. It is not an exhaustive
near-duplicate search, and it does not guarantee every physical room was identified
correctly. The test set uses **4,703** of the 5,050 official test images — the paper
must not say all official test images contribute questions.

---

## 6. Canonicalization

One implementation (`vocab.py`), shared by the generators and the evaluator:
lowercase and collapse whitespace → strip a trailing `[\d_\-]+` run (`wall23` →
`wall`) → singularize via `inflect`, with guards for `-ss` words and a plural-only
list → a hand-written **25-entry** synonym table → lookup in the **151-concept**
`canonical_objects.csv`.

At evaluation time `answer_form.canonical_answer_form` additionally strips trailing
`.!?`, removes a leading article, and maps number words to digits. `existence` and
`left_right` bypass the object vocabulary entirely. Matching is exact —
**deliberately not fuzzy**, so a genuinely different answer is never snapped onto a
similar-looking concept.

> **Implementation trap.** The vocabulary is keyed by *concept* (`tissuebox`) while
> the released `answer` column carries the *display name* (`tissue box`). So
> `canonicalize(...)["in_vocab"]` is `False` for every multiword concept. Any
> membership test must compare canonical **display forms** — testing `in_vocab`
> would flag 146 `trash can` and 48 `file cabinet` gold answers in the test split as
> out-of-vocabulary. `evaluate.py:is_legal_answer` documents this.

---

## 7. Structural limitations to disclose

Stated from what the rules can and cannot decide. No measured error rates.

1. **`existence` negatives are decided on canonical name alone.** A scene annotated
   `desk`, `counter`, or `coffeetable` yields gold `no` for `table`. The
   category-matching that makes a negative *plausible* is exactly what selects the
   confusable sibling, so the failure mode is built into the rule. Fixing it needs a
   conflicting-concept set at generation time — i.e. a rebuild, which this
   submission does not do.
2. **`existence` is dominated by one concept.** `table` is **74.0%** of test
   existence items (1,892 of 2,558; next is `bookshelf` at 240). The type's score is
   therefore largely a `table` score, and limitation (1) is concentrated in exactly
   that concept. Computed from the release, not from any audit.
3. **Out-of-frame referents are not excluded.** The intended rule ("more than 30% of
   the object outside the frame") is not computable from polygons, which cover only
   visible pixels. The gate exists in code but is inert in the released config.
4. **124 of 2,490 test `identify_superlative` rows bypass the margin.** When the
   depth-bearing pool holds exactly one object, the runner-up is `None` and no ratio
   test applies (`closest_camera` still enforces its 0.4 m floor). The blanket
   statement "requires the second-closest to be at least 1.2× the closest" is
   therefore not universally true — verified directly against the release.
5. **252 of 2,490 `identify_superlative` rows have winner and runner-up of the same
   concept.** Harmless for correctness, but the margin is not a uniqueness guarantee.
6. **Open-vocabulary rows carry an empty `answer_space`.** The build-time check that
   an answer lies in its declared space only runs on rows with a non-empty space, so
   `identify_superlative` and `nearest_object` answers are never validated against a
   closed set at build time.
7. **Pair-level rejections are not logged** for `left_right` and `relative_depth`, so
   the drop logs under-report filtering for those two types; only scene-level
   "nothing survived" reasons appear.
8. **`existence` negative evidence records no object index or geometry** — only the
   concept and a fixed reason string, so a negative item is not independently
   re-verifiable from its evidence field alone.
9. **The vocabulary is a committed input, not a reproducible artifact.** Re-deriving
   it yields 148 concepts against the shipped 151.
10. **The validation split is small** — 1,720 items, 340–352 per type.
11. **Depth-only answerability is a scope limitation**, not a claim that all indoor
    VQA is answerable from depth.

**Documentation discrepancy to fix in the manuscript:** `DATASHEET.md` states "up to
8 questions in train". Every type is deduped to at most one row per image and
`existence` to one row per image globally, so the structural maximum is **5**;
measured maxima are 5 (train) and **4** (val/test, verified directly).

---

## 8. Gate G1 status

| Phase 2 checklist item | Status |
|---|---|
| `freeze_release.py --verify v2.4` run and recorded | ✅ "verified, no drift" |
| Row counts and five per-type counts recorded | ✅ §1, §3 |
| Adapter reads released CSVs and preserves `question_id` | ✅ verified: `IDs` column equals release `question_id`, same order, zero duplicates, all three splits |
| No preprocessing script writes into the frozen release | ✅ verified: the only writers are `build_release.py` and `build_release_artifacts.py`, both rebuild-only (Path B) and forbidden for this submission. `evaluate.py` and `export_v1_schema.py` read only |
| Construction description per type | ✅ §3 |
| Source annotations, depth evidence, canonical answers, split procedure | ✅ §3–§6 |
| One worked example per type | ✅ §3 |
| Consistent RGB-teacher → depth-student direction in all text | ⬜ manuscript pass, not yet written |
| No v2.5, no row/split edits, audit untouched | ✅ nothing modified |

G1 closes once the two open rows are ticked. Neither blocks Phase 3.
