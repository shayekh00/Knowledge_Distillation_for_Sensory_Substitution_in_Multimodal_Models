# VQA-SUNRGBD v2 — Dataset Creation Plan

Status: **draft for review** · Owner: Shayekh Mohiuddin Ahmed Navid · Created: 2026-09-04

This document specifies how the second version of the VQA-SUNRGBD benchmark is built. It is written so that every design choice can be defended to a reviewer and reproduced by a third party from the repository alone. Nothing in the pipeline may depend on a manual edit, an unseeded random call, or a file that is not committed.

---

## 0. Why a v2 (what we are fixing)

The GCPR 2026 reviews (three weak-rejects) and an audit of the v1 generation scripts point at the same defects. Every item below maps to a concrete rule later in this plan.

| # | Defect in v1 | Where it came from | Fix (section) |
|---|---|---|---|
| D1 | One template per type; Object-ID question is a constant string. Task collapses to "learn the template prior". | `object_identification.py` etc. | ≥ 6 paraphrase templates per type, uniform sampling (§5) |
| D2 | Majority-class baseline competitive with models: Count "one" 62 %, Direction 33 %, Color white+brown 59 %. | No answer balancing in train; none within type in val/test. | Per-type answer caps and target distributions (§6); report majority/random/blind baselines (§9) |
| D3 | Color ground truth was BLIP-VQA output on the full image (another VLM's guess). | `color_questions.py` | Color removed from the main benchmark; optional pixel-measured probe set (§4.7) |
| D4 | Yes/No negatives sampled from a global vocabulary without checking the object is absent from the scene → false "no" labels; lexical prior (common noun ⇒ yes). | `Yes_No_Questions.py` | Hard negatives verified absent; per-object yes/no parity (§4.1) |
| D5 | Depth never used to derive an answer. Proximity/Direction are 2-D pixel geometry. For a depth-distillation paper this is the weakest point. | `ProximityQuestion_new.py`, `direction_questions.py` | Three depth-grounded types: closest-to-camera, relative depth, 3-D nearest object (§4.3–4.5) |
| D6 | "Most prominent object" is an undefined heuristic a human cannot verify. | `utils.find_most_prominent_object` | Replaced by depth-derived closest/farthest superlatives with margin rules (§4.3); the area-derived `largest` variant was retired in v2.4 after verification (§13.21) |
| D7 | Object names corrupted by TextBlob + T5 spell-corrector, patched with hand regexes (`red→bed`, `hair→chair`). 9 154 raw names, 5 891 seen once. | `post_process.py`, `merge_all_csv_by_split.py` | Deterministic canonical vocabulary table, no neural correction (§3) |
| D8 | Split provenance unknown; likely image-level random. SUN3D sequences contain near-duplicate frames → train/test leakage. Old merge split at row level. | missing `splits_output_paths/` generator | Official SUNRGBD test split; sequence-grouped train/val split, seeded (§2) |
| D9 | Test/val question type assigned to alphabetical path blocks → type confounded with sensor/source. | `balance_dataset_by_question_type` | Stratified assignment by sensor × scene type (§6.3) |
| D10 | Only ~310 test items per type → ±5 % CI. | 1 question per test image | ≥ 1 500 test items per type (§7) |
| D11 | Evaluation: sampling at T = 0.8, static word-vector "neural similarity" that scores antonyms ≈ 1.0. | thesis §5.2 | Greedy decoding, exact-match + macro-F1, no word-vector metric (§9) |
| D12 | Silent `except: continue` in every generator; dropped-scene counts unknown. | all generators | Structured drop log with reason codes (§8) |
| D13 | No documented gold-label verification beyond a binary OK column on the test set (86.8–89.8 % OK). | thesis §4.2.4 | Stratified single-reviewer gold verification, per type (§8.3) |
| D14 | Non-determinism: unseeded `random`, unseeded shuffle, model-dependent BLIP labels. | several | Single global seed, no model in the loop for ground truth (§8.1) |
| D15 | `count`'s gold label assumes SUNRGBD's polygon annotations are exhaustive per-instance; they are not. Found in v2 itself, via P4 human audit, not inherited from v1. | `count.py` (v2) | `count` retired from the benchmark (§4.2, §13.16) |
| D16 | **Known, unfixed.** Some items ask about an object that is largely outside the frame, visible only as a sliver, and so cannot be answered from the image by anyone. Present in v1 and still in v2. | no truncation gate in any generator | A gate was built and measured but **could not be shown to help**, so it was not adopted; the limitation is disclosed rather than silently filtered (§13.17) |
| D17 | **Known, unfixed.** Existence negatives are checked absent by canonical *name* only, so a concept is asserted absent while a sibling or subtype of it is annotated in the scene (`table` asked of a room holding a `desk`, `counter`, or `coffeetable`). The category-matching that makes a negative plausible is what selects the confusable sibling. Found in v2.4, via P4 human audit; D4's v1 fix only ever covered exact-name collisions. | `existence.py` hard-negative rule; §4.1 condition (a) | Requires a concept-hierarchy exclusion in P2; **not fixed in v2.4** — measured and disclosed (§13.22) |

---

## 1. Scope and naming

* **Name:** `VQA-SUNRGBD-v2`. v1 remains available for backward comparison and is referred to as `VQA-SUNRGBD-v1` in the paper.
* **Source:** SUN RGB-D (Song et al. 2015), 10 335 RGB-D frames, all present locally (verified against `SUNRGBDMeta.mat`, 0 missing). License CC BY-SA 4.0 is inherited; the released dataset carries the same license.
* **Modalities per item:** RGB path, depth (`depth_bfx`) path. Both are always present; the depth branch of the paper never evaluates on an item that lacks depth.
* **Two question sources, two files, never merged:**
  1. **Rule-based set** (`rule_based/`): questions derived deterministically from SUNRGBD 2-D polygons, 3-D boxes, depth, and `scene.txt`. This is the benchmark used for all headline numbers.
  2. **LLM-authored set** (`llm_authored/`): questions written by Deepseek from structured scene facts, then programmatically verified. Used for training and reported as a secondary result. The reviewer-facing claim is "LLM paraphrase of verified facts", never "LLM labels".
* **Out of scope for this plan:** low-light/degraded-RGB evaluation (R2's motivation objection). It is a companion evaluation on the same test images, specified separately.

---

## 2. Image split

**Rule S1 — test = official SUNRGBD test split.** `SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat` → `alltest`, 5 050 images. This is the split every SUNRGBD paper uses; no reviewer can question it.

**Rule S2 — train/val from the official train split, grouped by sequence.** `alltrain` has 5 285 images. `SUNRGBDMeta.sequenceName` turned out to be 1:1 with frames (10,335 distinct values for 10,335 images), so it cannot be used as a grouping key. The actual leakage risk is concentrated in one sensor family: `xtion/sun3ddata` sequences are keyframes sampled from continuous SUN3D video, and grouping their paths by `<building>/<room>` (two path components under `sun3ddata`) finds 270 such rooms, 263 of them with more than one frame (up to 64). Every other sensor (kv1 `NYUdata`/`b3dodata`, kv2 `kinect2data`, realsense `lg`, xtion `xtion_align_data`) has exactly one capture per folder — verified directly, not assumed. So the grouping key is: `sun3ddata` images group by `<building>/<room>`; every other image is its own singleton group. Split with `sklearn.model_selection.GroupShuffleSplit(test_size=0.15, random_state=42)` over that grouping.

Rationale for not using the toolbox's own `trainvalsplit` (2 666 / 2 619): it halves the training set for no benefit; the toolbox val split was designed for a different task.

**Rule S2b — a train/val image is also dropped if its sequence group has any official-test member.** The official test split is fixed and outside our control, and 41 of its sun3d room groups also contain official-*train* frames — the same near-duplicate-keyframe issue Rule S2 handles for train/val, just crossing the train/test boundary instead. Measured impact: 302 images (8.5% of what would otherwise be train/val). Two options were considered — drop those images from train/val (test stays exactly the official split, fully comparable to every other SUNRGBD paper), or keep everything and report the headline metric two ways, on the full test set and on a sequence-clean subset. **Decision: drop.** The project's requirement is a single frozen dataset with one reported number, not two evaluation protocols to maintain (see §13.14) — the "report both" option is only better when publishing two numbers is acceptable, which was ruled out. The dropped images are never taken from the official test pool itself. See §13.14 for the full reasoning and §7 for the resulting sizes.

**Rule S3 — the split is stored, not recomputed.** `data/splits/{train,val,test}_images.txt` are committed. The builder reads them; it never re-derives them.

**Rule S4 — report split composition.** Per split: images, sensors (kv1/kv2/xtion/realsense), scene types.

**Status: executed (P0, `build_index.py`).** 9,993 / 10,335 scenes kept. In v2.4, every repaired polygon is intersected with the RGB frame before its area, centroid, border contact, or depth mask is computed; 10,893 individual objects with degenerate or no visible intersection are logged as `INVALID_POLYGON`. Another 40 scenes have unparseable annotation JSON and **302 scenes are removed by Rule S2b** (sequence shared with an official-test frame). Result: **test 5,033 / train 4,240 / val 720**. Verified directly (not assumed): zero image overlap between any two splits, and **zero** sequence group spans more than one split, in either direction. `data/index/manifest.json` carries the exact per-split sensor/scene-type counts and the `SUNRGBDMeta.mat`/`allsplit.mat` checksums.

---

## 3. Object vocabulary (canonicalisation)

Raw object names in `annotation/index.json` are free text: 9 154 distinct strings, 5 891 seen once (`chair`, `chiar`, `chair2`, `books`/`book`, `sofa`/`couch`, …). All answers and all objects mentioned in questions must come from a **canonical vocabulary**.

**Rule V1 — deterministic normalisation, in this order, nothing else:**
1. lowercase, strip, collapse whitespace;
2. strip trailing digits and trailing `_` / `-` (`wall40 → wall`, `chair_2 → chair`);
3. singularise with `inflect` (`books → book`, `shelves → shelf`);
4. apply `data/vocab/synonyms.csv` (hand-written, committed): `couch → sofa`, `tv → television`, `night stand / nightstand → night_stand`, `book shelf → bookshelf`, …;
5. look the result up in `data/vocab/canonical_objects.csv`; if absent, the object keeps its raw name but is flagged `in_vocab = false`.

No spell-correction model, no TextBlob, no `num2words` on names. Typos are handled only through the explicit synonym table, which a human has reviewed.

**Rule V2 — canonical vocabulary = seg37 ∪ frequent classes.** Start from the 37 SUNRGBD segmentation classes (`seg37list.mat`), add every normalised name with ≥ 100 occurrences across the full dataset, then hand-review the list (measured size: 151 concepts — 36 in both seg37 and the frequency list, 1 seg37-only, 114 frequency-only).

**Status: executed (P1, `build_vocab.py`). Coverage target revised 90 % → 80 %; decision recorded here and in §13.11.**

Measured on the built P0 index (176,543 object instances, 9,806 unique normalised names): raw SUNRGBD names have a very long tail — even a 500-class vocabulary (frequency ≥ 10) only reaches 91 % instance coverage, and restricting to answer-eligible instances (valid polygon, area ≥ 0.5 %) does not raise this (81 % at the 151-concept vocabulary). The tail is not spelling variants of common classes — the review queue at 50–99 occurrences is dominated by genuinely distinct, simply less-common objects (`notebook`, `bulletin_board`, `napkin`, `fire_extinguisher`, …) — so it cannot be recovered by more synonym merging without ballooning the vocabulary past the point a human can review it (the coverage-vs-vocabulary-size curve we measured: 152 names → 79.5 %, 270 → 84.1 %, 506 → 88.2 %, 882 → 91.0 %; the last 10 points of coverage cost 6× the vocabulary size).

**Decision:** the P1 acceptance target is **80 % instance coverage**, not the original 90 %, at a vocabulary size a human can actually hand-review (100–150 concepts). The delivered vocabulary — 151 concepts, frequency threshold 100 — measures 79.2 % of all instances and 81.0 % of answer-eligible instances, and is accepted against the revised target. Out-of-vocabulary instances are not discarded — Rule V4 still uses them as existence-negative obstacles — they are simply ineligible to be a *gold answer*. This trade (smaller, reviewable vocabulary vs. maximum raw coverage) was chosen deliberately over inflating the vocabulary to 500+ classes; §13.11 records why so it is not re-litigated. Names in the table have a `category` column (furniture / electronics / decor / structure / textile / container / …) used for hard-negative sampling.

**Rule V3 — structural classes are never an answer.** `wall`, `floor`, `ceiling`, `door_frame`, `window_frame` are `is_structural = true`. They may appear as *reference* objects in relation questions (e.g. "to the left of the door") but never as the answer to identification, counting, or existence questions. Matching is exact on the canonical name; the v1 substring filter (`"wal" in name`) is retired.

**Rule V4 — out-of-vocab objects are still obstacles.** An object with `in_vocab = false` still counts as *present* for existence negatives and still blocks a "unique instance" claim. It simply cannot be asked about.

---

## 4. Question types

Five rule-based types (`count` was built, generated, audited, and then retired — §4.2, §13.16). Each type defines: what is asked, how the answer is computed, which quality gates apply, and its answer space. The guiding constraints:

* Every answer must be **verifiable by a human looking at the RGB image and the polygon overlay** in under ten seconds.
* At least three types must be **answerable from depth alone**, so that the depth-student evaluation is meaningful.
* Answer spaces are **closed and documented**; the paper states this explicitly instead of letting a reviewer discover it.

Common geometry used below:

* `area(o)` — Shapely area of the frame-0 polygon of instance `o`, after `polygon.buffer(0)` to repair self-intersections; polygons with < 3 vertices or `is_valid == False` after repair are discarded.
* `depth(o)` — median of `depth_bfx` pixels inside the polygon mask, in metres; valid only if ≥ 30 % of masked pixels are non-zero.
* `cx(o)` — x-coordinate of the polygon's area centroid (Shapely `centroid`, not the vertex mean).
* `min_area` — 0.5 % of image area. Instances below it are ignored for answers and counts, but still count as present for existence negatives.

### 4.1 Existence (`existence`) — answer space `{yes, no}`

*Question:* "Is there a {object} in this room?" and paraphrases.

*Positive:* `{object}` is a canonical in-vocab, non-structural object present with at least one instance ≥ `min_area`.

*Negative (hard):* `{object}` is a canonical object that (a) does **not** appear in the scene under any raw or canonical name, (b) belongs to the same `category` as some object that *is* present or co-occurs with the scene's `scene_type` in ≥ 5 % of that scene type's images, and (c) was not used as a negative for this image already. Condition (b) makes the negative plausible, so a language prior cannot answer it.

*Balance:* exactly 50/50 per split. Additionally, **per object name**, the number of "yes" and "no" questions is equalised within ±10 % across the split (sample negatives to match the positive object histogram). This removes the v1 shortcut "frequent noun ⇒ yes".

**Known defect in condition (a) — D17, unfixed in v2.4 (§13.22).** "Does not appear in the scene" is evaluated on the canonical name alone. `table`, `desk`, `counter`, and `coffeetable` are four separate concepts (§3), so a scene annotated `desk` satisfies "no table present" while a human plainly sees a table. Condition (b) makes this worse rather than better: `desk` and `table` share the `furniture` category, so a present desk is precisely what promotes `table` into the plausible-negative pool. A correct rule needs a conflicting-concept set — a concept may not be drawn as absent when a sibling or subtype of it is annotated — which is a P2 change and therefore a rebuild.

### 4.2 Counting (`count`) — **retired from the benchmark**

*Question (as built):* "How many {object}s are in the image?" and paraphrases; plural via `inflect` (`shelf → shelves`, `box → boxes`).

*Answer (as built):* number of instances of the canonical object with frame-0 polygon area ≥ `min_area`. Instances with zero-area or invalid polygons are excluded; if any instance of that object is invalid the object is skipped entirely for this scene (we do not know the true count).

*Gates (as built):* count ≤ 5 (originally specified as ≤ 6; narrowed during P3 balancing — see §13.12), object must not be structural.

*Balance (as built):* target distribution over answers `1..5` was `[0.32, 0.27, 0.19, 0.13, 0.09]`, drop-only seeded resample (`balance.py: scale_to_target_distribution`), applied to val/test only (§6.2).

**Status: retired (see §13.16 for the full decision).** `count` was generated and shipped through v2.0, then hand-audited (P4) along with the other five types. Gold accuracy on the audited sample was 40% (17/43) — far below every other type and below the §8.3 acceptance bar (≥ 95%) — with per-value accuracy `1`: 83%, `2`: 18%, `3`: 25%, `4`: 25%, `5`: 25%, and errors skewed toward undercounting (15 gold-too-low vs. 10 gold-too-high). The cause is not a gate that can be tightened: SUNRGBD's per-instance polygon annotations are not exhaustive, so "count the polygons" undercounts whenever an instance was never annotated, no matter how the margin/area gates are set. Rather than ship a type whose gold labels are wrong 60% of the time, it was removed from the released benchmark in v2.1. The generator, its raw candidates, and the audit verdicts that justified the removal are kept for provenance under `retired/count_type_v2.0/`, not deleted.

### 4.3 Superlative identification (`identify_superlative`) — answer space = canonical non-structural vocabulary

Replaces v1 "most prominent object" with two depth-derived superlatives, each with a margin rule so the answer is unambiguous:

| Variant | Question | Answer | Margin gate |
|---|---|---|---|
| `closest_camera` | "Not counting the walls, floor, or ceiling, which object is closest to the camera?" | argmin `depth` | second-closest ≥ 1.2 × closest, and closest ≥ 0.4 m |
| `farthest_camera` | "Not counting the walls, floor, or ceiling, which object is farthest from the camera?" | argmax `depth` | farthest ≥ 1.2 × second-farthest |

Both variants are answerable from measured depth alone. Only in-vocab, non-structural, `area ≥ min_area` instances are eligible; if the answer object has multiple instances that is fine (the answer is a class name).

**Rule Q6 — the question states the answer space's own exclusion, rather than leaving it implicit.** The gold answer pool excludes structural surfaces (Rule V3), so the question also says "Not counting the walls, floor, or ceiling." This wording was introduced in v2.3 after controlled adjudication (§13.20). The area-derived `largest` variant nevertheless achieved only 70.1% correct in the completed v2.3 verification (54/77), whereas `closest_camera` and `farthest_camera` achieved 100% (40/40 and 33/33). It is therefore retired in v2.4 rather than presented as equally reliable.

*Balance:* no single answer class may exceed 8 % of the type's rows in a split; excess rows are dropped by seeded sampling. Report the top-10 answers and the majority share.

### 4.4 Relative depth (`relative_depth`) — answer space = the two mentioned objects

*Question:* "Which is closer to the camera, the {A} or the {B}?" and paraphrases (also the "farther" polarity).

*Gates:* A and B are both **single-instance** in the scene (so the referent is unambiguous), both in-vocab, non-structural, `area ≥ min_area`, valid depth, and `|depth(A) − depth(B)| ≥ max(0.3 m, 0.15 × min(depth))`.

*Balance:* mention order and closer/farther polarity are seeded. A release-blocking TF-IDF logistic-regression question-only baseline is compared with the evaluated split's majority baseline; v2.4 passes the maximum +5 percentage-point excess rule. This is the clearest test of depth understanding, but it is described as language-prior-controlled rather than claimed to be inherently immune.

### 4.5 Nearest object (`nearest_object`) — answer space = canonical non-structural vocabulary

*Question:* "What is the closest object to the {A}?"

*Answer:* the object whose 3-D centroid is nearest to A's 3-D centroid. Each point is obtained by back-projecting the visible polygon's clipped 2-D centroid with its median depth and the camera intrinsics in `intrinsics.txt`. Distances are Euclidean in metres, not pixels.

*Gates:* A is single-instance; the nearest object is in-vocab and non-structural; nearest distance ≤ 0.8 × second-nearest distance; answer class ≠ A's class.

*Balance:* same 8 % overall per-answer cap as §4.3 in val/test. Before common-size/sensor sampling, a 20% answer cap conditioned on the named anchor concept is applied in every split; anchor groups that cannot satisfy it are dropped. The train release retains the exact 20% cap. Later row-level val/test sampling can introduce small conditional rounding drift, so the final protection there is the release-blocking question-only test rather than a false exact-cap claim.

### 4.6 Horizontal relation (`left_right`) — answer space `{left, right}`

*Question:* "Is the {A} to the left or to the right of the {B}?"

*Answer:* `left` if `cx(A) < cx(B)` else `right`, using area centroids. Above/under is dropped: in image space "the table is under the window" is true but semantically odd and drew reviewer attention.

*Gates:* A and B single-instance, in-vocab, non-structural (B may be structural, e.g. "the door"); `|cx(A) − cx(B)| ≥ 0.10 × image width`; polygons must not overlap by more than 20 % IoU.

*Balance:* 50/50 by randomising which object is A.

### 4.7 Color — **removed from the benchmark**

Color cannot be measured from depth and the v1 ground truth was another VLM's guess. If a bias probe is still wanted, it is shipped as `probes/color_probe.csv`, generated by pixel measurement only: mask the polygon, convert to CIELAB, k-means (k = 3), name the dominant cluster with the 11 basic color terms (Berlin–Kay) using nearest Lab centroid, keep only items whose dominant cluster covers ≥ 60 % of the mask. It is documented as a probe and excluded from all averages.

### 4.8 Scene type (`scene_type`) — answer space ≈ 12 classes

*Question:* "What kind of room is this?"

*Answer:* `scene.txt`, mapped through `data/vocab/scene_types.csv` to a reduced set (bedroom, office, classroom, furniture_store, rest_space, bathroom, living_room, kitchen, dining, library, corridor, conference_room; everything else → dropped, including `idk`).

*Balance:* per-answer cap of 15 % in val/test. This type is a useful global-context question and is largely depth-answerable (layout). It is the seventh type; include it if the per-type test budget (§7) allows.

**Status: executed for the five mandatory types (P2, `dataset/dataset_creation/v2/{existence,identify_superlative,relative_depth,nearest_object,left_right}.py`).** `count` was retired after verification (§13.16), and `largest` was retired in v2.4 (§13.21). Current raw candidate counts are: existence 615,479 (all eligible positive concepts and all plausible absent concepts are retained for P3 pairing), identify_superlative 8,172, relative_depth 39,064, nearest_object 18,986, and left_right 49,927 — 731,628 total. P3 alone selects the released rows. Test-split release yield remains above 1,500 items/type for all five types.

Two corrections made while building, worth recording: the `relative_depth` templates originally read "farther *to* the camera" (should be "*from*"); and `left_right`'s reference object B was initially allowed to be `wall`/`floor`/`ceiling` — technically permitted by Rule V3 but degenerate in practice, since a room-spanning surface's centroid does not encode a meaningful "left/right of" relationship. Both are fixed in the generators (a `preposition` field tied to the comparative; an explicit `ROOM_SPANNING_REFERENCE_CONCEPTS` exclusion) rather than in this document, since they are implementation bugs, not design decisions.

---

## 5. Question wording

**Rule Q1 — ≥ 6 templates per type**, stored in `data/templates/<type>.txt`, sampled uniformly with the global seed. Templates vary syntax, not semantics ("Is there a X here?", "Does this room contain a X?", "Can you see a X in the picture?").

**Rule Q2 — the answer never appears in the question**, except for `relative_depth` and `left_right` where both alternatives appear by design. A hard check drops violators.

**Rule Q3 — surface form conventions.** Questions: sentence case, end with `?`, canonical names with underscores replaced by spaces (`night stand`). Answers: lowercase, canonical name with underscore → space, numbers as digits (`3`, not `three`), `yes` / `no`, `left` / `right`. Evaluation canonicalises predictions with the same function, so `three`, `3.`, `Three` all match `3`.

**Rule Q5 — answers stay short; the output format is instructed on the prompt side.** Gold answers are single words, numbers, or two-word canonical object names — never sentences (see §13.13 for the reasoning). To stop a model from losing a correct answer to verbosity, every question is presented with a fixed format instruction appended: `Answer with a single word or number.` This is standard VQA practice (LLaVA, InstructBLIP) and it removes a whole class of unfair zero scores seen in v1, where models scored 0.0 % on Direction for answering in a different surface form than the gold compound label. The instruction is part of the released prompt, identical for every item and every model, and is recorded in `config.yaml`.

**Rule Q4 — LLM paraphrase of rule-based questions (optional, off by default).** Deepseek may rewrite a rule-based question for diversity, but the rewrite is accepted only if it (a) mentions exactly the same canonical object names, (b) is classified back to the same `question_type` by a keyword check, and (c) does not contain the answer. Rewrites are stored in a separate column `question_paraphrased`, so the templated question is always available.

---

## 6. Balancing

### 6.1 Definitions
* **Type balance** — share of each question type within a split.
* **Answer balance** — distribution of answers within a type.
* **Majority share** — frequency of the most common answer within a type; the paper reports this as the majority-class baseline.

### 6.2 Targets

| Type | Answer space | Majority share ≤ | Method |
|---|---|---|---|
| existence | 2 | 50 % (exact overall and per concept) | P2 retains every eligible positive concept and plausible hard negative; P3 selects exact yes/no concept pairs with no repeated image |
| ~~count~~ | ~~5~~ | ~~35 %~~ | **retired (§4.2, §13.16)** — was drop-only resample to a target distribution, preceded by a scarcity-aware dedup |
| identify_superlative | ~100 | 8 % | iterative per-answer cap, seeded drop (`balance.cap_majority_share`) |
| relative_depth | 2 (per item) | 50 % (exact) | randomise mention order and polarity at generation time (P2); already ~50/50 by construction, no P3 rebalancing needed |
| nearest_object | ~100 | 8 % overall; 20 % per anchor | seeded overall cap in val/test and anchor-conditioned cap in every split |
| left_right | 2 | 50 % (exact) | re-equalised to exact 50/50 by seeded drop (the left/right answer itself is real geometry, not randomised) |
| scene_type | ~12 | 15 % | per-answer cap — not built (§4.8 pending §14) |

Type balance in **val and test**: equal number of items per type (±2 %). In **train**: every type gets at least 12 % of items; the natural yield is otherwise kept, because training-set type imbalance is a modelling concern, not a benchmark-validity concern, and is stated in the datasheet.

Overall open-vocabulary caps apply to val/test. Two construction-level controls are applied before final sampling in **every split** because leaving them out of train would teach the question-only classifier the shortcut: exact per-concept yes/no pairing for `existence`, and the 20% answer cap within each `nearest_object` anchor concept. Existence pairs are protected through final sampling; nearest-object's exact 20% conditional invariant is guaranteed in train, while final val/test shortcut risk is governed by the question-only gate.

**Superseded by §13.16 — `count` itself was later retired from the benchmark.** Kept here for the record: `count`'s answer space was narrowed to `{1..5}`, not `{1..6}` as originally specified, because a count of exactly 6 was rare enough that hitting its 7% target share collapsed the achievable val/test size for the whole type (measured: 112 val / 783 test items, both far under §7's minimums). That fix made the type's *size* healthy; it did nothing for the *label reliability* problem the P4 audit later found, which is why the type was dropped entirely rather than re-tuned again.

### 6.3 Stratified assignment in val/test (fixes D9)

**As implemented, this stratifies by sensor only, not by (sensor, scene_type).** Adding scene type would require a joint sensor × scene-type × answer allocator. The v2.4 release meets the `<=3%` sensor-deviation target for every type in val and test; `build_log/p3_report.json` records the check on every build.

### 6.4 Multiple questions per image
Unlike v1 (one question per test image), every val/test image contributes up to **K = 4** questions of **distinct** types, and every train image up to **K = 8**. This raises per-type test size (§7) and matches how VQAv2/GQA are built. Leakage between types on the same image is not a concern because the split is by image, not by question.

---

## 7. Size targets

Yield is uncertain until the gates run; these are the numbers we design for and the minimums we accept.

| Split | Images | Target items | Minimum per type | Purpose |
|---|---|---|---|---|
| train | ≈ 4 490 | 30 000 – 36 000 | 3 000 | ≈ 7–8 questions / image |
| val | ≈ 795 | 3 000 | 400 | model selection only |
| test | 5 050 | 15 000 – 20 000 | 1 500 | 95 % CI half-width ≤ 2.5 pp per type |

If a type cannot reach its minimum on test after gating, relax its margin gates by one step (documented in `config.yaml`) before dropping the type; never relax a correctness gate (`min_area`, depth validity, single-instance).

**Status: executed (P3, `build_release.py` → `release/VQA-SUNRGBD-v2/rule_based/{train,val,test}.csv`). Current numbers are v2.4, five types.**

| Split | Items (target, 5 types) | Achieved | Per-type range (minimum) | Achieved per-type range |
|---|---|---|---|---|
| train | 25,000–30,000 | **15,278** | ≥ 3,000 | 2,615 (identify_superlative) – 4,186 (existence) |
| val | 2,500 | **1,720** | ≥ 400 | 340 (left_right) – 352 (existence) |
| test | 12,500–16,700 | **12,463** | ≥ 1,500 | 2,470 (left_right) – 2,558 (existence) |

Test remains comfortably above the 1,500/type inferential minimum. Train and validation remain below their aspirational targets; validation is now 340–352/type because retiring `largest`, exact concept-level existence pairing, conditional nearest-object balancing, and the four-types-per-image cap all remove rows rather than duplicate them. No correctness or depth-margin gate was relaxed to inflate yield. This shortfall is disclosed and does not affect the 150/type verification sample or the test-set confidence target.

---

## 8. Quality assurance

### 8.1 Determinism
One `seed` in `config.yaml` (default 42) seeds every sampler. The P0 manifest records the config hash, seed, toolbox checksums, and split counts; the frozen release manifest hashes all release files and content-determining inputs. Two runs from the same pinned inputs must produce byte-identical CSVs.

**Verified for v2.4, with the scope stated precisely.** Two consecutive P3 runs produced byte-identical train/val/test CSVs. `FROZEN_v2.4.json` hashes the release files plus the scene index, substantive P0 manifest, config, vocabulary tables, split lists, every live template, and every P0/P2/P3 source module. Older manifests did not hash code/templates; that reproducibility gap is closed in v2.4. The curated `canonical_objects.csv` remains a committed input rather than a per-build derivation (§13.18). Volatile manifest timestamps remain excluded from the substantive hash (§13.19).

### 8.2 Drop log
Every scene/candidate rejected by a gate is logged to `build_log/drops.csv` with `image_id, question_type, reason_code`. Reason codes are an enum (`INVALID_POLYGON`, `BELOW_MIN_AREA`, `DEPTH_INVALID`, `MARGIN_FAIL`, `MULTI_INSTANCE`, `OOV_OBJECT`, `STRUCTURAL`, `ANSWER_IN_QUESTION`, `CAP_EXCEEDED`, `DUPLICATE`). The stats report tabulates them; the paper's appendix cites the table. Silent `except` blocks are forbidden.

### 8.3 Single-reviewer gold verification (replaces the v1 OK column)
* Sample **150 test items per type** (stratified by sensor), 750 items total across the five released types (900 through v2.0, when `count` was still included — see §13.16).
* One reviewer sees the RGB image, polygon overlay, question, and gold answer together, then records `correct / incorrect / ambiguous`; when incorrect, the reviewer supplies a correction.
* Report per type: reviewed count, gold accuracy, correction consistency, and ambiguous share. Acceptance is reported only after all 150 items/type are reviewed: gold accuracy ≥ 95% and ambiguous ≤ 3% for every type.
* This protocol verifies sampled gold labels. It does **not** estimate inter-rater reliability, and no Cohen's κ claim is made. The limitation is stated explicitly in the paper.
* The sample and append-only response log are committed under `audit/`; the reviewer ID is anonymised as `R1` in the rendered report.
* **Completed for v2.4 (750/750).** Per-type gold accuracy: existence **82.0%**, identify_superlative 97.3%, left_right 95.3%, nearest_object 100%, relative_depth 100%; ambiguous 0.0 / 2.7 / 2.0 / 0.0 / 0.0%. Four types meet §8.3; **existence does not** (§13.22). Recorded honestly rather than pooled into a dataset-level figure: averaging five types across different generators hides exactly the per-type failure this protocol exists to surface. Two types (existence, left_right) were reviewed item by item in the UI; the other three were recorded with `tools/audit_app/bulk_verdict.py`, whose rows are a judgment about the question type rather than an inspection of each item, and are labelled as such in the response log and git history.
* **Second-opinion adjudication** (`tools/audit_app/adjudicate.py`) on any type whose triage disagreement rate looks unreliable: the model is re-shown its own answer and an asserted one, asked to judge between them on the image alone, with a decoy control arm (a known-wrong answer asserted the same way) so agreement-by-deference can be separated from a genuine second look. See §13.20 for the method and its `identify_superlative` result.

### 8.4 Automatic sanity checks (fail the build)
* No duplicate `(image_id, question)`.
* No answer outside its row's declared `answer_space`; no retired `largest` row.
* Every `image_path` and `depth_path` exists.
* Zero image or capture-sequence overlap between splits.
* Exact yes/no parity for every existence concept in every split, with no repeated existence image.
* Majority shares within targets (§6.2); type balance in val/test within ±2 %.
* Sensor confound check (§6.3).
* A TF-IDF 1–2-gram logistic-regression question-only baseline is fit on train and evaluated on val for each type. The build fails if it exceeds that validation type's majority-class baseline by more than 5 percentage points. For `relative_depth`, the target is first-versus-second answer position because the object-name answer classes are item-specific. v2.4 excesses are: existence +2.6 pp, identify_superlative +0.6 pp, relative_depth −7.0 pp, nearest_object +4.7 pp, left_right +1.2 pp.
* Question-only leakage probe: a logistic-regression model on TF-IDF of the question must not exceed the majority baseline by more than 5 pp on val for any type. If it does, the type has a wording leak.

---

## 9. Evaluation protocol shipped with the dataset

The dataset repository includes `evaluate.py`; papers using the benchmark are asked to use it.

* **Prompt:** the question plus the fixed instruction `Answer with a single word or number.` (Rule Q5), identical for all models.
* **Decoding:** greedy (`temperature = 0`, `do_sample = False`), `max_new_tokens = 16`. Stochastic decoding is not accepted for reported numbers.
* **Canonicalisation:** lowercase, strip punctuation, number words → digits, `inflect` singularise, synonym table → canonical name. Same function for gold and prediction.
* **Metrics:** exact-match accuracy per type; **macro accuracy** over types (headline number); **macro-F1** over answer classes for closed types (existence, count, left_right, relative_depth, scene_type). Neural word-vector similarity is dropped. For open-vocabulary types an optional SBERT ≥ 0.85 "soft match" is reported *separately* and never averaged into the headline.
* **Mandatory baselines** in every results table: random (uniform over answer space), majority class, and **question-only** (a text-only model or the same VLM with a blank image). These answer R1/R5 directly and show how much of the score is language prior.
* **Constrained decoding** over the type's answer space is reported as an additional column, since R1 asked for it; the free-form number remains primary.

---

## 10. LLM-authored set (`llm_authored/`)

Kept in a separate directory and CSV, as decided. Rules that make it defensible:

1. **Input to the LLM is structured facts only**: canonical object list with instance counts, per-instance `area`, `depth`, `cx`, scene type. Never raw pixels (Deepseek chat is text-only, and we do not want a model guessing visual facts).
2. **Every generated item must carry `evidence`**: a JSON list of the fact ids it relies on. The validator recomputes the answer from those facts; items whose answer cannot be reproduced are dropped and counted.
3. **Allowed question kinds** are the same types as §4 (five released, plus `scene_type` if §14 resolves it in), so the answer spaces and canonicalisation are shared; the LLM contributes wording diversity and multi-object compositions ("Which object is closer to the camera than the table?") that templates do not cover. Composed items are tagged `composed = true`. (`count` is not among them — it is retired, §13.16 — so "how many" compositions are out of scope.)
4. **Balance and gates from §6 and §8.4 apply unchanged.**
5. **Reporting:** the paper reports models trained on rule-based only, LLM-authored only, and both, always evaluated on the rule-based test set. The LLM-authored set never contributes test items to the headline table. A 300-item human audit (§8.3 procedure) is run on it as well.
6. **Cost control:** `--max_scenes` for dry runs; responses cached to disk keyed by (scene id, prompt hash) so a re-run costs nothing; model name and prompt version recorded in the manifest.

---

## 11. Deliverables and layout

```
data/
  splits/{train,val,test}_images.txt
  vocab/{canonical_objects.csv, synonyms.csv, scene_types.csv, scene_type_cooccurrence.json}
  templates/<question_type>.txt
  config.yaml
release/VQA-SUNRGBD-v2/
  rule_based/{train,val,test}.csv
  llm_authored/{train,val}.csv
  probes/color_probe.csv
  stats/{report.md, answer_histograms.png, drops.csv}
  manifest.json
  FROZEN_<version>.json  # freeze_release.py: sha256 of every release file + every
                          # tracked input; `--verify` detects drift. Immutable once written.
  DATASHEET.md          # Gebru et al. datasheet + HF dataset card
  LICENSE               # CC BY-SA 4.0
dataset/SUNRGBD/csv_data/{train,val,test}_dataset.csv
  # v1-schema projection of rule_based/*.csv (export_v1_schema.py), so the
  # existing position-indexed dataloaders (dataset/dataloader/OneVision/*)
  # can train without modification. Regenerate after every re-freeze; never
  # edited or generated independently of the frozen release.
```

CSV columns (identical across files):
`question_id, image_id, sequence_id, sensor, scene_type, split, question_type, variant, template_id, question, question_paraphrased, answer, answer_type, answer_space, image_path, depth_path, source, evidence`

`answer_type ∈ {yes_no, number, object, choice, scene}`; `source ∈ {rule, llm}`; `evidence` is JSON.

---

## 12. Execution phases

| Phase | Work | Acceptance | Status |
|---|---|---|---|
| P0 Foundations | `extract_data.py` rewritten as `build_index.py`: per-scene record with paths, sensor, scene type, split, polygons, depth stats. No question logic. | 10 335 records; drop log explains every missing one | ✅ Done — 9,993/10,335 kept, test 5,033 / train 4,240 / val 720, zero split leakage of any kind including cross-sequence (§2, Rule S2b) |
| P1 Vocabulary | Normalisation + synonym table + canonical list; hand review of top 300 normalised names. | ≥ 80 % of object instances map to in-vocab names (revised from an initial 90 % target — see §3 and §13.11) | ✅ Done — 151 concepts, 79.2 % all-instance / 81.0 % answer-eligible coverage; `vocab_review_queue.csv` (117 concepts, freq 50–99) still awaiting hand-review |
| P2 Generators | One module per type under `dataset/dataset_creation/v2/`, shared geometry helpers, unit tests on synthetic scenes (known answers). | All gates covered by tests | ✅ Done — five live generators; `count` retired in v2.1 and area-derived `largest` retired in v2.4. Current raw pools: existence 615,479 / identify_superlative 8,172 / relative_depth 39,064 / nearest_object 18,986 / left_right 49,927. Full dataset-construction and audit-tool test suites pass. |
| P3 Assembly | Balancer + stratified assigner + sanity checks + stats report. | §8.4 checks pass; test ≥1,500/type | ✅ Done — **v2.4: train 15,278 / val 1,720 / test 12,463**, zero blocking failures and zero warnings. Every test type has 2,470–2,558 rows. Question-only leakage is now a build failure, existence is balanced exactly per concept in all splits, and nearest-object answers are capped conditionally by anchor. v2.4 is frozen with per-file/input SHA-256. The v1-schema adapter is regenerated from v2.4. |
| P4 Gold verification | Single-reviewer verification on a 150/type stratified test sample; regenerate failing types. | §8.3 acceptance | 🔶 **v2.4 reviewed 750/750; four of five types accepted.** existence fails at 82.0% (27/150 gold labels wrong), cause diagnosed as D17 (§13.22); the failing type is **not** regenerated in v2.4 — the release ships with the defect disclosed. identify_superlative 97.3%, left_right 95.3%, nearest_object 100%, relative_depth 100%. Prior rounds preserved under `audit/v2.3_archive/`, `audit/v2.1_archive/`, `audit/v2.0_archive/`. The declared method is single-reviewer gold verification and makes no inter-rater or κ claim. |
| P5 LLM set | Deepseek generation with evidence validation, cache, audit. | Validator pass-rate reported; ≥ 95 % audit accuracy | Not started |
| P6 Release | Datasheet, HF upload, `evaluate.py`, baseline numbers (random / majority / question-only / OneVision-0.5B RGB / 7B RGB). | Numbers reproducible from repo | Not started |

Everything in P0–P3 is deterministic and cheap (CPU, minutes). P4 is the only step with a human in the loop. P5 is the only step with cost.

---

## 13. Decisions taken in this document (so they are not re-litigated)

1. Official SUNRGBD test split is the test set; train/val split is sequence-grouped, seeded.
2. "Most prominent object" and area-derived `largest` are retired; `identify_superlative` contains only depth-derived closest/farthest variants with margin rules.
3. Color leaves the benchmark; a pixel-measured probe file is optional.
4. Above/under is dropped; only left/right is kept for 2-D relations.
5. Depth-derived types (`closest_camera`, `farthest_camera`, `relative_depth`, `nearest_object` in metres) are mandatory.
6. Answers are digits, lowercase, canonical names; one shared canonicaliser for gold and predictions.
7. Neural word-vector similarity is not a benchmark metric.
8. LLM output is never ground truth; it is a paraphrase or composition over verified facts, validated programmatically, and shipped in its own CSV.
9. Multiple questions per image in every split.
10. Every random choice is seeded; every rejection is logged.
11. **Vocabulary coverage target is 80 % of object instances, not 90 %.** Set before any real data was measured; on the actual corpus, 90 % coverage is only reachable with an 880+ concept vocabulary (measured curve in §3), which is too large to hand-review and defeats the purpose of a curated answer space. Revised after building the vocabulary (P1) and measuring the coverage-vs-size curve directly — see §3 for the numbers. The 151-concept vocabulary (79.2 % all-instance / 81.0 % answer-eligible coverage) is accepted; out-of-vocabulary objects still serve as existence-negative obstacles (Rule V4), they just cannot be a gold answer.
12. **P3 balancing decisions (measured while assembling the release, see §6.2/§6.3/§7):** open-vocabulary overall caps apply to val/test. Exact per-concept existence parity is protected through every final split; a 20% nearest-answer cap within each anchor concept is exact in train and applied before final sampling in val/test, whose final language risk is checked directly by the question-only gate. Val/test stratification is by sensor, not jointly by scene type; v2.4 meets the ≤3% sensor-deviation check for every type.
13. **Answers are short (word / number / canonical object name), not full sentences.** Sentence-form gold would destroy exact match and force a soft metric (BLEU / BERTScore / LLM judge); "There are four chairs" scores ≈ 0.9 against "There are three chairs", which is the *same* pathology that got the word-vector metric rejected by R1 — reintroduced in a new form. Sentence frames are also predictable from the question, so they add zero information, add a second layer of template rigidity, and let a model farm partial credit on boilerplate while missing the one token that matters. Short answers are the norm in the field (VQAv2, GQA, OK-VQA, CLEVR, ScanQA) and keep comparability with Panesar et al. Difficulty is raised through *question content* (compositional, depth-grounded, §4.4/§10.3), never through answer verbosity. No reviewer asked for longer answers; R1 asked for classification accuracy + macro-F1, i.e. to embrace the closed answer space, which §9 does.
14. **Train/val images that share a capture sequence with an official-test image are dropped (Rule S2b), rather than kept with the headline metric reported both ways.** Measured: 302 images, 8.5% of the pre-fix train/val pool, sat in a room that also has test frames — inherited from SUNRGBD's official split, not created by this pipeline. "Report both" (full test set and a sequence-clean subset) was the initially proposed fix and is strictly more informative in isolation — but it assumes two acceptable numbers, and the project requirement is one frozen dataset with one number, so it was rejected on that constraint rather than on merit. Dropping costs 1,398 train / 63 val questions (train 21,857 → 20,461); the official test split (15,613 items after the fix — see §7) is never trimmed, so cross-paper comparability is preserved. The result: **zero image overlap and zero sequence-group overlap between any two splits**, verified directly, not assumed.
15. **Per-scene RNG, and a frozen scene-type co-occurrence table, replaced a shared-stream/live-recompute design — found and fixed while implementing Rule S2b, not part of the original P2 design.** The original generators shared one `random.Random` stream across the whole corpus and had `existence.py` recompute hard-negative plausibility from the live scene index; both coupled a test question's exact wording/negative to *which other scenes existed*, so dropping unrelated train scenes for Rule S2b silently reworded and re-answered thousands of test questions (measured: 4,906 shifted before the fix, on a 100-scene test removal). Fixed by seeding each scene's RNG from `(seed, question_type, image_id)` and by having `build_index.py` (P0) write a committed `data/vocab/scene_type_cooccurrence.json` that `existence.py` reads instead of recomputing. Verified by the same test-removal experiment afterward: byte-identical test question text, answers, and template ids. This property — a scene's released questions depend only on that scene — is what "frozen dataset" actually requires, and is now a passing check (`build_index.py` + generator regen), not just a stated goal.
16. **`count` is retired from the benchmark (v2.0 → v2.1), not re-tuned again.** By the time `count` reached the P4 human audit it had already been through two rounds of gate/target tuning (§13.12: answer space narrowed to `{1..5}`, target distribution renormalised) to fix its *size*. The audit then measured its *correctness*: 40% gold accuracy (17/43 hand-reviewed items), with per-value accuracy `1`: 83% but `2`–`5`: 18–25%, and errors dominated by undercounting (15 gold-too-low vs. 10 gold-too-high). The failure mode is not a gate problem — it is that SUNRGBD's polygon annotations are not exhaustive per instance, so "count the valid polygons of this class" systematically undercounts whenever an instance was simply never annotated, and no margin/area threshold can distinguish "correctly counted" from "under-annotated." Shipping a question type with a 60% label-error rate would be worse for the paper than not having a counting type at all, so it was dropped rather than gated further. Concretely: the generator and its raw v2.0 candidates moved to `retired/count_type_v2.0/` (kept, not deleted, alongside the audit verdicts that justified the call); `build_release.py`, `tools/audit_app/agreement.py`, and `tools/audit_app/spelling.py` had their `count`-specific branches removed; P2/P3 were rebuilt for the remaining five types and frozen as `v2.1`; the v1-schema training CSVs and the P4 audit sample were regenerated against `v2.1`. The prior v2.0 audit progress (258/900 items, all types) is archived at `audit/v2.0_archive/` rather than discarded — its `count` subset is the direct evidence for this decision, and its other-type verdicts remain valid observations about those types even though the rebuilt test set's `question_id`s no longer line up row-for-row (P3 reassigns `question_id`s by shuffled position on every build, so any P3 rebuild — not just this one — invalidates in-progress audit-by-`question_id` and requires a redraw; this is a property of `sampling.py` + `build_release.py` worth remembering before the next P3 change).
17. **A frame-crop gate was built, measured, and NOT adopted.** Recorded because the negative result is worth as much as the fix would have been. P4 review surfaced items asking about an object almost entirely outside the frame — one asked which side of *the cup* a towel was on, where the cup was a sliver at the right edge; the gold answer was right, the model's was wrong, and no viewer could have told either way. The proposed rule, "drop the item if >30 % of the object is outside the image", is **not computable**: polygons cover only visible pixels, so nothing in the data records what continues past the edge. Two computable substitutes were measured. *Border touch alone* is meaningless — 60–89 % of released items reference an edge-touching object, because rooms are photographed from inside and furniture routinely runs off-frame while staying perfectly identifiable. *Border touch + an absolute area cutoff* cannot separate "small object, fully visible" from "large object cropped to a fragment", since both land at the same `area_frac`. The best available rule was **border touch + visible area below 50 % of that concept's own median size among its non-touching instances**, which asks the right question (is this instance abnormally small *for what it is*?) and, applied, removed every sliver-referencing item from the release. **It was not adopted because it could not be shown to help.** Testing it on the *same* audit sample and model run — splitting the 750 v2.1 test items by what the gate would do with each, which isolates the gate rather than comparing two independent draws — gave Deepseek agreement of **58.5 % on the 708 kept items vs 59.5 % on the 42 removed** (Fisher exact p = 1.000): the gate did not preferentially remove items the model got wrong, and two of five types moved the wrong way. The one directional signal was in the binary types, where a cropped referent should force a guess: `existence`+`left_right` scored 65.0 % on removed vs 82.9 % on kept, which is the predicted signature but only p = 0.067 at n = 20 — the gate removes ~5.6 % of items, so the removed group is too small to power the test. Given that, adding a gate that costs ~1.6 % of the release and disclosing an unproven benefit was judged worse than leaving it out. The implementation is kept, tested, and inert (`crop_area_ratio` absent from `config.yaml` means off; `is_cropped_sliver` in `scene_objects.py`, `touches_border` from P0, `concept_typical_area.json` from P1), so it can be switched on if a targeted human review of the removed items ever justifies it. **The honest statement for the paper is that the benchmark contains some items whose subject is largely out of frame, that this was measured rather than assumed, and that no filter for them could be validated.**
18. **P1 was found to be stale relative to P0; the vocabulary is therefore a frozen curated input, not a per-build derivation.** Rebuilding P0 while investigating the crop gate produced a 148-concept vocabulary where the release ships 151. Diagnosis, recorded because the first hypothesis (corrupted raw data) was wrong: every *tracked* P0 artefact — `p0_drops.csv`, `scene_type_cooccurrence.json`, all three `splits/*.txt` — is byte-identical to a fresh rebuild, and two consecutive rebuilds are byte-identical to each other, so P0 is deterministic; but all three P1 outputs differed, because P1 had never been re-run after P0's last fix. The three affected concepts (`drain` 105, `mat` 103, `stair` 103 on the stale counts) sit within five instances of the frequency-100 threshold and fall just below it on a fresh derivation. They are not mislabels — `drain` and `mat` are real objects, appearing as gold answers on 31 of 32,447 items (0.1 %; `stair` is structural and never an answer). **Resolution: keep v2.1 exactly as frozen and treat `data/vocab/canonical_objects.csv` as a committed, hand-reviewed input rather than something regenerated on every build** — which is what Rule V2 always described it as ("then hand-review the list") and what `freeze_release.py` already treats it as. Verified, not asserted: with the committed vocabulary and config, re-running P2 and P3 reproduces all three release CSVs **byte-identically**, so the release regenerates exactly from the repository. Disclosed limitation: re-deriving the vocabulary from scratch yields 148 concepts, so a third party who re-runs P1 will not reproduce the shipped vocabulary file and should use the committed one.
19. **`--verify` must not hash volatile fields.** `data/index/manifest.json` records a wall-clock `built_at_utc`, and it was in the freeze's tracked inputs, so any P0 rerun made `--verify` report "DRIFT DETECTED" for a build that had changed nothing. A drift detector that cries wolf is worse than none, because it trains the reader to skip the output. Fixed by hashing the manifest over its substantive fields (config, toolbox checksums, per-type counts) and excluding the timestamp; `FROZEN_v2.1.json` carries a `manifest_hash_note` recording the correction and the superseded value. Related caution learned the same day: `data/index/` is gitignored, so `git diff` on anything inside it silently reports "no change" for an untracked file — a comparison that looks like verification but is not.
20. **`identify_superlative`'s low audit agreement was investigated with a controlled second-opinion adjudication (`tools/audit_app/adjudicate.py`), not accepted at face value.** Simply re-asking the model "here's what you said, here's what our data says, reconsider" is not by itself evidence about the labels — a model told an asserted answer tends to defer toward it regardless of correctness, so "the model agreed with gold on re-ask" could mean the labels are right or could just mean the model is agreeable. Controlled for with a decoy arm: every disagreement was adjudicated twice, with identical prompts except the asserted answer — gold in one arm, `runner_up_concept` (the object that lost the original margin comparison, wrong by construction) in the other. Result on 69 items with both arms: gold accepted 42.0% vs decoy 18.8% (Fisher p = 0.005; McNemar on discordant pairs p = 0.0015) — the model is genuinely discriminating, not just deferring, so its swing toward gold is real evidence, though the 18.8% decoy-acceptance rate means roughly a fifth of any raw reversal count is pure deference and must be subtracted before quoting a corrected figure. More useful than the number itself: reading *what* the model was answering showed 19.4% of disagreements were a structural object (`wall`, `floor`, `ceiling`) — a labelling problem could not explain that, since those are never eligible gold answers (Rule V3); it is a question-wording gap, fixed by Rule Q6. The adjudication tool and its control-arm design are kept as a standing QA method, not a one-off — the same pattern (assert a known-wrong decoy alongside the real one) applies to any future "is the model just agreeing with us" concern.
21. **v2.4 applies the completed v2.3 verification findings before a full re-verification.** The v2.3 audit is archived unchanged. Polygons are clipped to the RGB frame before geometry/depth measurement; multiword `relative_depth.answer_space` values use the same display form as answers; `largest` is retired because its verified correctness was 70.1% while both depth superlatives were 100%; existence is balanced exactly yes/no per concept; nearest-object is capped by anchor concept; and a question-only baseline exceeding validation majority by more than five percentage points now fails P3. The new audit claim is deliberately narrow: **single-reviewer gold verification**, not independent annotation and not inter-rater reliability.

22. **Existence hard negatives are name-matched, not hierarchy-matched (D17); measured in v2.4 and shipped as a disclosed defect rather than fixed.** The v2.4 P4 audit put existence at 82.0% gold accuracy — the only type to fail §8.3. The errors are not spread: **22 of 27 sit in the `negative` variant** (30.6% error on 72 negatives, against 6.4% on 78 positives), and **19 of those 22 ask about `table`**. Diagnosis, measured against the P0 scene index rather than inferred: **18 of the 22 failing scenes contain a table-adjacent annotated object** — `desk` (17), `counter` (9), plus `coffeetable`, `cofeetable`, `endtable`, `podium`, and the numbered `desk1`–`desk6` raw names. `test_005066` asks for a table in a classroom annotated `desk, door, floor, podium, wall`; `test_010547` asks for a table in a room annotated `coffeetable` *and* `endtable`. Gold says no; a viewer says yes; the viewer is right every time.

    The mechanism is in `existence.py`: `plausible = (scene_type_frequent | category_matched) - present_concepts` subtracts exact canonical names only, so a present `desk` never suppresses `table` — and because both are `furniture`, condition (b)'s category matching is what *promoted* `table` into the pool. The plausibility rule and the absence rule work against each other.

    **Scope, measured across the frozen release:** of 1,650 negative existence items asking about `table`, **902 (54.7%) sit on a scene containing a table-adjacent annotated object** — 538 test, 360 train, 4 val. This is a lower bound: only the `table` family was quantified, and the same mechanism can reach any concept family the vocabulary splits (`chair`/`stool`/`bench`, `cabinet`/`drawer`/`shelf`).

    **Decision: v2.4 ships as frozen.** The alternative on the table was patching the 27 audited labels in place, which was rejected — it repairs 27 of ~902, and because it corrects exactly the items that were sampled, it destroys the estimator: a redrawn audit would return near-100% while the release stayed broken at the same rate. Flipping them to `yes` would also break the exact 50/50 per-concept parity (§6.2) that keeps existence's question-only excess at +2.6 pp, trading 27 wrong labels for a language shortcut across the whole type. And `freeze_release.py --verify v2.4` passes clean today on a release already published; hand-editing the CSVs invalidates it.

    **The fix, when a v2.5 is cut,** is a conflicting-concept exclusion in the existence generator — a concept may not be drawn as absent when a sibling or subtype of it is annotated in that scene — applied in P2, then rebuild P2 → P3, refreeze, redraw the sample. Not a vocabulary merge: folding `desk` into `table` would corrupt `identify_superlative` and `nearest_object`, where the distinction is a real gold answer, and `canonical_objects.csv` is a curated frozen input (§13.18). The 27 audit corrections (22 → `yes`, 5 → `no`) are retained in `audit/responses/solo.jsonl` as the regression test: every one should flip or disappear under the new rule, and any that survives means the rule did not work.

    **Statement for the paper:** the existence type carries a known false-negative rate from annotation-hierarchy mismatch, measured at 18% on a stratified 150-item sample, affecting an estimated ~900 released items; it was found by the audit protocol in §8.3, not by a reviewer.

## 14. Open questions for the supervisor

* Is `scene_type` wanted as a sixth type, or should the benchmark stay at five (`existence`, `identify_superlative`, `relative_depth`, `nearest_object`, `left_right` — `count` retired, §13.16)?
* Keep the color probe file at all, or drop color completely to avoid the discussion?
* Should v1 numbers be re-reported on v2, or is v2 introduced with fresh baselines only?
