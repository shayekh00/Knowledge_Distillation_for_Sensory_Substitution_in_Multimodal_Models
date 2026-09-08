# VQA-ARKitScenes v1 — second-dataset plan

**Status:** proposal, awaiting author decision.
**Date:** 2026-09-07.
**Purpose:** answer reviewer comments R1.5, R2.1 and R5.2 — the only weakness all
three reviewers raised independently — by adding a second, genuinely independent
benchmark built from **ARKitScenes** (Apple, NeurIPS 2021 Datasets & Benchmarks),
reusing the existing question-generation pipeline.

---

## 1. Why ARKitScenes, and why not NYUv2

**NYUv2 was ruled out on evidence.** SUN RGB-D is a composite dataset assembled
from NYU Depth v2, Berkeley B3DO and SUN3D. Measured against our frozen v2.4
release, `SUNRGBD/kv1/NYUdata/` accounts for 2,660/15,278 train rows (17.4%),
334/1,720 val (19.4%) and 1,760/12,463 test (14.1%) — **1,427 of NYUv2's 1,449
labeled images, 98.5%, are already in our benchmark, spread across all three
splits.** Using it as a second dataset would mean evaluating on images we train
on, and NYUv2's own 795/654 split does not align with ours, so its test images
sit in our train split. Recording this here because it is a durable fact about
the benchmark that anyone reading these results needs to know.

**ARKitScenes is disjoint from SUN RGB-D and answers a stronger question.**

| Property | Value |
|---|---|
| Scale | ~5,000 captures over ~1,660 unique scenes |
| RGB | iPad Pro wide camera; `lowres_wide` frames are depth-aligned |
| **Depth** | **LiDAR time-of-flight** (plus Faro laser-scanner ground truth on a subset) |
| Annotations | 3D oriented bounding boxes, 17 furniture categories |
| Splits | Official `Training` / `Validation` |
| Licence / access | CC BY-NC-SA; direct download, **no approval loop** |

Two reasons this beats the alternatives:

1. **Different sensor physics.** SUN RGB-D is entirely structured-light and
   stereo (Kinect v1/v2, Xtion, RealSense). ARKitScenes is LiDAR ToF. **R1 named
   LiDAR explicitly** among the modalities they wanted. A result holding across
   structured light *and* ToF is a materially stronger "sensory substitution"
   claim than a second structured-light dataset. ScanNet — the better pure
   engineering fit — is the same sensor family as what we already have, so it
   would demonstrate scene generalization but *not* modality generalization.
2. **It is available today.** ScanNet requires a signed terms-of-use agreement
   emailed for approval. Requesting ScanNet access in parallel costs nothing and
   is recommended, but nothing should be blocked on it.

**The honest cost:** ARKitScenes ships 3D boxes over 17 classes, not 2D masks
over a rich vocabulary. §4 covers what that does to each question type, and §7
states the resulting limitations plainly. This is the real trade for the two
advantages above, and it is more engineering than a mask-based dataset would be.

---

## 2. The pipeline seam we build against

`build_index.py` (P0) is the **only** component touching SUN RGB-D formats
(`SUNRGBDMeta.mat`, `allsplit.mat`, the toolbox layout). Verified by inspection:
`existence.py`, `left_right.py`, `nearest_object.py`, `identify_superlative.py`
and `relative_depth.py` contain **zero** references to SUN RGB-D paths, `.mat`
files, or the toolbox — they read `data/index/scene_index.jsonl` and nothing
else.

So the work is **one new indexer emitting the same schema**; vocabulary, question
generation, balancing, answer-form normalisation, release building, freezing,
training and evaluation are all reused unchanged.

Target schema (one record per frame):

```
image_id, sensor, scene_type, image_width, image_height,
rgb_path, depth_path, annotation_path, split, sequence_id,
objects: [ { object_index, raw_name, is_valid_polygon,
             area_px, area_frac, centroid_x, centroid_y,
             depth_median_m, depth_valid_frac, touches_border } ]
```

Thresholds in `data/config.yaml` must be honoured or consciously re-declared —
`geometry.min_area_frac: 0.005`, `depth.min_valid_fraction: 0.30`,
`depth.clip_max_m: 8.0`, `seed: 42`. Changing any of them changes the manifest
hash, which is the point.

One additional dependency: `nearest_object.py` calls
`depth_utils.load_intrinsics(scene_dir)`, which reads a per-scene
`intrinsics.txt`. The ARKitScenes indexer must write an equivalent file per
frame or scene, since ARKitScenes intrinsics are per-frame
(`lowres_wide_intrinsics/`).

---

## 3. The central technical problem: 3D boxes, not 2D masks

Our per-object fields are 2D image measurements. ARKitScenes gives 3D oriented
boxes in world coordinates plus per-frame camera poses (`lowres_wide.traj`) and
intrinsics. Each object's 2D fields must therefore be **derived by projection**:

1. Transform the 3D OBB's 8 corners into the camera frame using the frame's pose.
2. Reject boxes wholly behind the camera or outside the frustum.
3. Project surviving corners with the frame's intrinsics; take the 2D convex hull
   (`shapely` is already a dependency, used for SUN RGB-D polygons).
4. Clip the hull to the image rectangle; compute `area_px`, `area_frac`,
   `centroid_x/y`, `touches_border` from the clipped hull — exactly the
   reductions `build_polygon_records` already performs.
5. Compute depth statistics inside the hull from the LiDAR depth map, giving
   `depth_median_m` and `depth_valid_frac` on the same definitions as now.

**Occlusion is the hard part and must be handled, not ignored.** A projected 3D
box lands in the image whether or not the object is actually visible — a chair
behind a wall still projects. An unfiltered indexer would generate "is there a
chair?" for frames where no chair is visible, which is a *wrong label*, not merely
a noisy one. Proposed test, to be validated rather than assumed:

> For each projected object, compare the observed LiDAR depth inside the hull
> against the box's own depth range in camera frame. If the observed depth is
> substantially closer than the box across most of the hull, something occludes
> it — mark the object not visible and exclude it from answer eligibility.

This mirrors how `depth_valid_frac` already gates SUN RGB-D objects, so it fits
the existing `eligible` concept rather than adding a new one.

**A genuine upside:** for `relative_depth` and `nearest_object`, ARKitScenes'
3D boxes give *true metric geometry*, which is more reliable than SUN RGB-D's
"median depth over a hand-drawn polygon". Depth-ordering questions should be
**more** trustworthy on this dataset, not less. Whether to define depth from the
3D box centroid or from the observed LiDAR median inside the hull is a decision
to declare explicitly (recommendation: observed median, for consistency with
v2.4's definition; record the box-centroid value alongside as evidence).

---

## 4. Per-question-type impact of the 17-class vocabulary

| Type | Impact | Assessment |
|---|---|---|
| `existence` | Works; needs the occlusion test to be right | Fine |
| `left_right` | Centroid comparison from projected hulls | Fine |
| `relative_depth` | True 3D geometry available | **Better than v2.4** |
| `nearest_object` | Answer space is the 17 classes | Workable, more repetitive |
| `identify_superlative` | "Largest" from projected-hull area is coarser than mask area | Weakest; consider 3D box volume as an alternative definition, declared up front |

The 17 classes (cabinet, refrigerator, shelf, stove, bed, sink, washer, toilet,
bathtub, oven, dishwasher, fireplace, stool, chair, table, TV/monitor, sofa) are
comparable to the NYUv2-40 and ScanNet-20 benchmark vocabularies, and our answer
space is already closed and canonicalised — so this is a limitation to state, not
a blocker.

---

## 5. Disk and download budget

**Constraint: 43 GB free** (`/data`, 98 GB total, 55% used; `dataset/` is
currently 16 GB). The full ARKitScenes 3DOD subset is far larger than that, so
**a bounded subset must be selected up front.**

The key efficiency: **we do not need many frames per scan.** Consecutive frames
are near-duplicates, and `sequence_id` grouping would place them in the same
split group anyway. v2.4 draws 15,278 questions from 4,187 images. So sampling
roughly **5–10 well-spaced frames from each of ~500 scans** yields 2,500–5,000
images — comparable scale — at a small fraction of the download.

Procedure:
1. Use the official `download_data.py` with `--video_id_csv` to fetch a seeded
   random sample of Training/Validation scans, in batches.
2. After each batch, subsample frames (fixed stride or pose-diversity based),
   keep only the retained `lowres_wide` / `lowres_depth` / intrinsics / pose
   entries, and delete the rest before fetching the next batch.
3. Record the exact `video_id` list and frame indices in the release manifest so
   the subset is reproducible — this is part of the frozen identity, like
   `tracked_inputs` in `FROZEN_v2.4.json`.

---

## 6. Phased plan

### Phase 0 — Feasibility probe (half a day, no GPU)
Download **3–5 scans only**. Verify: pose and intrinsics parse; a projected 3D
box lands where the object visibly is (eyeball 10 frames); the occlusion test
separates visible from occluded instances; LiDAR depth decodes to plausible
metres. **Gate: if projection or occlusion cannot be made to work on 5 scans,
stop and switch to ScanNet** rather than scaling up a broken indexer.

### Phase 1 — Indexer (2–3 days, no GPU)
`dataset/dataset_creation/v2/build_index_arkit.py`, emitting the §2 schema.
Deliverables: unit tests mirroring `test_freeze_release.py`'s discipline,
including a **negative control** (a deliberately occluded object must be excluded)
and a projection round-trip test on synthetic geometry with known answers.

### Phase 2 — Subset download (~1 day, mostly waiting)
Per §5, to a declared budget of ≤ 25 GB, leaving headroom.

### Phase 3 — Generate and freeze VQA-ARKitScenes v1 (half a day, CPU)
Run the unchanged P2 generators, then `build_vocab`, `balance`, `answer_form`,
`build_release`, `freeze_release.py`. Produces a frozen release with its own
manifest and sha256s under the same discipline as v2.4.

### Phase 4 — Human audit (half a day)
Reuse `tools/audit_app/` — the existing sampled-audit tool with evidence
overlays. Audit ~200 questions across the five types. **This matters more here
than it did for v2.4**, because projected boxes plus an occlusion heuristic is a
weaker labelling path than hand-drawn polygons, and R1/R2 already criticised
label quality. A measured per-type error rate from this audit is itself a
reportable result and pre-empts the obvious attack.

### Phase 5 — Reduced ladder (~25–30 GPU-hours)
Not the full matrix — the rows that carry the claim: `B1`, `B2` (zero-shot depth
and RGB), `B3` (depth CE), `B5` (RGB CE), `X2` (CE+KD, unaligned), and
`D0`→`D3` plus `D0`→`D5` (aligned, CE-only and CE+KD). Eight rows, single seed
(§5 policy), same 10-epoch/patience-2 budget. A new teacher feature cache and
top-K logits cache are required for this dataset (~2 minutes and ~35 minutes
respectively, at measured rates).

### Phase 6 (free, once Phase 5 exists) — Cross-dataset transfer
Evaluate the SUN RGB-D-trained D5 checkpoint on VQA-ARKitScenes and vice versa.
Pure inference, no training. This is the strongest single answer to R5.2's
"cross-dataset evaluation could be more beneficial", and it costs almost nothing
once both benchmarks exist.

**Total: roughly 1–1.5 weeks calendar**, dominated by the indexer and the audit.

---

## 7. What this buys, and what it does not

| Reviewer point | Status after this plan |
|---|---|
| R1.5 single benchmark | **Addressed** — second, independent, human-annotated dataset |
| R1.5 modality generalization | **Addressed** — structured light *and* LiDAR ToF |
| R5.2 other datasets / cross-dataset | **Addressed** — Phase 6 gives true cross-dataset transfer |
| R2.1 more than one foundational model | **Not addressed** — needs the cross-family student track (SmolVLM2); separate effort |
| Templated questions (R1.3, R2.4) | **Not addressed** — questions remain template-generated |

**Stated limitations, to appear in the manuscript rather than be discovered by a
reviewer:**

1. **Coarser object extents.** Projected 3D boxes over-estimate 2D extent versus
   segmentation masks, which most affects `identify_superlative`.
2. **Occlusion is heuristic.** Visibility comes from a depth-consistency test,
   not ground-truth masks. Phase 4's audit must quantify the residual error rate.
3. **17-class vocabulary** narrows `nearest_object` and `identify_superlative`
   relative to v2.4.
4. **Still templated.** A second dataset answers "generalizes across scenes and
   sensors", not "generalizes beyond template questions". Answering the latter
   needs human-written or human-verified questions — scope separately.

---

## 8. Decisions needed

1. Proceed with Phase 0 (3–5 scans, half a day) as a go/no-go gate?
2. Request ScanNet access in parallel now, as insurance and as a possible third
   dataset?
3. `identify_superlative` on ARKitScenes: projected-hull area (consistent with
   v2.4) or 3D box volume (more accurate, but a different question)? Must be
   declared before generation, not chosen after seeing results.
4. Run the leave-one-source-out study (train excluding `kv1/NYUdata`, test on it)
   as an interim generalization result? It needs no download and ~8 GPU-hours,
   and can proceed while Phases 0–2 run.
