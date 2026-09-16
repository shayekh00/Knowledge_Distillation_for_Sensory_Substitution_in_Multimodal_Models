# M³FD Thermal-VQA Dataset Plan

**Status:** approved design; execution plan only. No M³FD data, release, or
training result exists yet.

## 1. Goal and Claim Boundary

Build an additive `VQA-M3FD-Thermal-v1` benchmark from M³FD's registered RGB
and thermal image pairs. The training-time teacher may receive RGB, but the
student receives exactly one thermal image and the question at inference.

The resulting study tests RGB-to-thermal transfer for object and 2-D spatial
VQA. It does **not** test metric geometry, depth reasoning, arbitrary sensory
substitution, or general modality transfer.

The frozen VQA-SUNRGBD-v2.4 release, its artifacts, and its reported results
must remain byte-for-byte unchanged.

## 2. Source and Feasibility Gate

M³FD contains 4,200 registered RGB--infrared pairs with six detection classes:
`person`, `car`, `bus`, `motorcycle`, `truck`, and `lamp`. It is an object
detection source, so its boxes are useful evidence for visible objects and
2-D geometry but not for metric depth relations.

Before downloading or indexing the full source, run a bounded, stratified
probe. Record the source URL, licence/terms, download checksum, original file
layout, annotation schema, image dimensions, and source version in a manifest.

The probe must verify all of the following on at least 100 pairs stratified by
class and scene condition:

1. Each RGB image has exactly one thermal partner and one annotation record.
2. The source box coordinate system is identified unambiguously.
3. Source boxes can be represented in thermal-frame pixels and visibly cover
   the same objects in both modalities.
4. A capture-session or equivalent grouping key can be recovered to prevent
   temporal and near-duplicate leakage.

Construction proceeds only if at least 95% of manually inspected mappings are
valid. If pairing, registration, or capture-group provenance cannot be
established, do not publish a random-frame M³FD VQA release; document the
failure and select another source instead.

## 3. Dataset Contract

### 3.1 Release layout

Create a separate release:

```text
release/VQA-M3FD-Thermal-v1/
  rule_based/{train,val,test}.csv
  FROZEN_v1.0.json
  DATASHEET.md
  stats/
```

Keep M³FD-specific transient inputs outside the frozen release:

```text
data/index/scene_index_m3fd.jsonl
data/vocab_m3fd/
data/candidates_m3fd/
build_log/m3fd/
```

Each release row follows the existing candidate/release contract and includes
`question_id`, `image_id`, `sequence_id`, `split`, `question_type`, `question`,
`answer`, `answer_type`, `answer_space`, `source`, and JSON `evidence`.

Use explicit image fields:

- `rgb_path`: source RGB image relative to the project dataset root.
- `thermal_path`: source infrared image relative to the project dataset root.

Do not reuse `depth_path` to store a thermal path. The index retains original
and thermal-frame box coordinates, source annotation path, class, dimensions,
capture group, and pairing metadata.

### 3.2 Split contract

Use a deterministic 70/15/15 train/validation/test split with seed `42`.
The unit of splitting is a verified capture session or scene group, never an
individual frame. Build a perceptual-hash near-duplicate index for RGB and
thermal imagery as an independent check.

The release must fail if any image ID, capture group, or detected near-duplicate
group crosses splits. If source grouping cannot be verified, the release is
blocked rather than silently using a frame-random split.

### 3.3 Student input contract

Add a dedicated `m3fd` dataset branch and a `thermal` modality option to the
training and inference loaders. In thermal mode, every student input must come
from `thermal_path`; it must not read RGB paths, teacher caches, annotation
files, boxes, or answer metadata. RGB teacher mode reads only `rgb_path`.

Existing `sunrgbd` and `arkitscenes` behavior remains unchanged.

## 4. Question Design

Questions are deterministic templates grounded only in M³FD annotations. No
LLM produces a gold label. Each candidate stores the classes, boxes, margins,
template ID, and rule that produced its answer in `evidence`.

Only these four types are in v1:

| Type | Gold evidence and inclusion rule | Answer space |
|---|---|---|
| `existence` | A class has at least one valid visible annotated box; a negative class is plausible but absent from complete source annotations. | `yes|no` |
| `count` | Count valid visible boxes of one class, only when the source annotation is verified complete for that class/frame. | Declared number words in the retained count range. |
| `left_right` | Two unambiguous boxes have IoU at most 0.20 and centroid horizontal separation of at least 10% of thermal-frame width. | `left|right` |
| `identify_superlative` | The largest clipped thermal-frame box has area at least 1.20 times the runner-up. | The six-class canonical vocabulary. |

Exclude relative depth, nearest-object, color, above/below, image-region, and
scene-level questions. M³FD boxes cannot establish those claims reliably.

Use at least six frozen templates per type. Preserve `template_id` in every
row. Generate candidates with a per-image deterministic random seed derived
from the global seed, question type, and image ID, matching the existing
reproducibility convention.

### 4.1 Balance and shortcut controls

- Balance existence and left/right exactly 50/50 within every split.
- Balance existence positives and negatives within each object class.
- Cap the majority count answer and superlative class at 35% in validation and
  test; record the achieved distribution rather than fabricating rows to reach
  a target.
- Fit random, training-majority, and TF-IDF question-only baselines on train
  only. Block the release when TF-IDF exceeds the majority baseline by more
  than five percentage points for a type.
- Keep at most one candidate per `(image_id, question_type)` in validation and
  test. This prevents one scene from dominating the reported macro score.

## 5. Implementation Phases

### Phase 0 -- Probe and provenance

1. Obtain M³FD under its research terms and store source provenance.
2. Implement a read-only inspection script for pair matching, XML/annotation
   parsing, image dimensions, class counts, and source naming patterns.
3. Render RGB and thermal overlays for the 100-pair stratified registration
   review.
4. Determine the source grouping key. If no defensible grouping exists, stop.

### Phase 1 -- Index and vocabulary

1. Add `build_index_m3fd.py` that emits the shared scene-object schema plus
   `thermal_path`, modality-specific boxes, and capture-group provenance.
2. Clip boxes to image bounds; reject empty, non-finite, duplicate, tiny, or
   failed-registration objects with reason-coded logs.
3. Add `data/vocab_m3fd/canonical_objects.csv` with exactly the six published
   classes and a small explicit synonym map.
4. Write unit tests for source parsing, pairing, clipping, coordinate mapping,
   canonicalization, and invalid-record handling.

### Phase 2 -- Split and candidate generation

1. Add a group-aware M³FD split assignment command with seed `42` and a split
   manifest.
2. Add M³FD-specific candidate output locations without changing the SUN
   RGB-D candidate files.
3. Implement the four generators and their deterministic templates.
4. Emit reason-coded candidate drops for every rejected frame or object.
5. Test all positive and negative conditions, spatial margins, count rules,
   answer spaces, determinism, and split isolation.

### Phase 3 -- Build, evaluate, and freeze

1. Extend release construction for `dataset=m3fd` and its four-type macro
   contract; do not alter VQA-SUNRGBD-v2.4 construction rules.
2. Extend `evaluate.py` with the M³FD vocabulary and release-directory
   overrides required for canonical exact-match, per-type accuracy, macro-F1,
   invalid-output rate, and question-only baselines.
3. Extend training and inference interfaces with `--dataset m3fd` and
   `--modality thermal`.
4. Add inference-isolation tests proving thermal inference cannot open RGB or
   annotation files.
5. Run an end-to-end smoke build from a small probe subset before the full
   release.

### Phase 4 -- Audit and release gate

Audit 150 stratified test examples per retained type against the RGB image,
thermal image, source annotation, and generated overlay. A type is retained
only when gold accuracy is at least 95% and ambiguity is at most 3%.

Existence and count require an explicit annotation-completeness finding. If
either fails audit, remove that type from v1 before freezing; do not keep it
with a warning. The final release must contain at least 200 validation and 200
test rows for every retained type.

Once all gates pass, generate the datasheet, per-type statistics, drop reports,
input/output hashes, and immutable `FROZEN_v1.0.json`. Verify the release using
the project's freeze verification command before any model training.

## 6. Required Tests

- Pairing, annotation parsing, class normalization, and invalid-record tests.
- RGB/thermal coordinate-mapping and box-clipping tests.
- Deterministic group split, zero-leakage, and near-duplicate tests.
- Existence, count, left/right, and superlative generation tests.
- Release balancing, answer-space, question-ID, and macro-metric tests.
- Thermal-only loader tests that deny RGB, teacher-cache, and annotation access.
- Freeze/verification test and a full probe-release integration test.

## 7. Acceptance Criteria

The dataset is ready for model experiments only when:

1. The Phase 0 feasibility gate passes.
2. Every release split is capture-group and near-duplicate disjoint.
3. Every retained question type passes audit and has at least 200 validation
   and 200 test instances.
4. Shortcut baselines and required release checks pass.
5. Thermal-only inference isolation passes.
6. `FROZEN_v1.0.json` verifies with no drift.

No claim about RGB-to-thermal knowledge distillation is made until a matched
thermal CE baseline and the selected KD recipe are trained and evaluated under
the frozen contract.
