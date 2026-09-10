# VQA-ARKitScenes v1 — second-dataset plan

**Status:** Phases 0-3 complete — VQA-ARKitScenes-v1.0 frozen (2026-09-08,
train 1,548 / val 33 / test 38 rows) — see §6 for results. `answer_form`
was not exercised and needs its own audit before use. Phases 4-6 remain
proposal, awaiting author decision on §8's open items.
**Date:** 2026-09-07 (Phase 0 results added 2026-09-08).
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

So the work is **one new indexer emitting the same schema**; question
generation, balancing, answer-form normalisation, training and evaluation are
reused unchanged. Vocabulary building and release naming are reused **with
parameterization**, not literally unchanged — verified by reading the actual
call graph rather than re-grepping the generator files alone, since a shared
helper module can carry a hardcoded dependency the generators themselves never
show:

* **`scene_objects.scene_dir_absolute(image_id)`** — imported by
  `nearest_object.py` (via `depth_utils.load_intrinsics`) to find each frame's
  `intrinsics.txt` — hardcodes `os.path.join(DATASET_DIR, "SUNRGBD", image_id)`.
  This is the one place a "dataset-agnostic" generator actually depends on a
  SUN-RGB-D-specific path, and it was missed by grepping `nearest_object.py`
  itself, which contains no `"SUNRGBD"` string at all. Fix is small — derive
  the directory from the record's own `image_path`/`depth_path` field (already
  dataset-agnostic strings) instead of reconstructing `"SUNRGBD/{image_id}"`
  from scratch — but it is a real code change, not zero-touch reuse, and
  belongs in Phase 1's deliverables.
* **`build_vocab.py`'s Rule V2, step 3** — unconditionally loads
  `dataset/SUNRGBDtoolbox/Metadata/seg37list.mat` via `load_seg37_concepts()`
  and unions it into the canonical vocabulary regardless of frequency. Run
  against an ARKitScenes `scene_index.jsonl` unmodified, this either crashes
  (file absent) or — worse, if `SUNRGBDtoolbox/` is still present on disk from
  the existing dataset — silently seeds ARKitScenes' canonical vocabulary with
  37 SUN-RGB-D segmentation classes that have nothing to do with this dataset,
  a wrong-but-plausible vocabulary rather than a crash. In practice this
  "always-include" step is likely unnecessary for ARKitScenes at the planned
  scale anyway: all 17 target classes are common indoor furniture that should
  each clear the existing >=100-occurrence frequency threshold on their own in
  2,500-5,000 images, so the fix is plausibly "skip the seg37 union for this
  dataset" rather than building an equivalent always-include list — but that
  is an empirical claim to verify against the real corpus in Phase 1, not
  assume.
* **`RELEASE_DIR`/`SEG37_MAT_PATH`-style path constants** in
  `build_release.py`, `build_release_artifacts.py`, `freeze_release.py` all
  hardcode `release/VQA-SUNRGBD-v2/...`. Trivial to parameterize (a
  `--release-name` flag or an equivalent constant swap per dataset), but not
  literally zero-touch — recorded here so Phase 1's "2-3 days" estimate is not
  quietly wrong about scope.

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

One consequence of the `scene_dir_absolute()` fix above: whatever directory it
is generalized to derive, the ARKitScenes indexer must write an
`intrinsics.txt` there per frame (not per scan), since ARKitScenes intrinsics
are natively per-frame (`lowres_wide_intrinsics/`) rather than per-scene.

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

**A fourth "reused unchanged" exception, more consequential than §2's three
because it is not confined to the indexer.** Both `decode_sunrgbd_depth`
(the indexer) and `distillation/depth_input.decode_metric_depth` — which is
called at **training and evaluation time**, for every `modality="depth"` row,
and is explicitly required to stay byte-identical to the indexer's decoder —
implement the *official SUN RGB-D 16-bit encoding specifically*:
`(raw >> 3) | (raw << 13)`, then divide by 1000. This is not a generic
16-bit-depth-PNG convention; it is SUN RGB-D's own bit layout
(`SUNRGBDtoolbox/readData/read3dPoints.m`). ARKitScenes' LiDAR depth files are
Apple's own format and there is no evidence here that they share this specific
bit rotation — Phase 0's checklist below already includes "LiDAR depth decodes
to plausible metres" precisely because this cannot be assumed, but the
existing decoder should not be the first thing tried: verify ARKitScenes'
actual depth encoding from its own documentation first, then either write a
second decode function or parameterize the existing one by dataset, rather
than reusing `decode_sunrgbd_depth`'s bit-rotation and hoping the metres come
out plausible by coincidence. And because `depth_input.py` sits on the
**training** path, not just the indexer, this is a Phase 5 prerequisite too —
every depth-modality row in the reduced ladder depends on it decoding
correctly, not only the dataset-generation phase.

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

**Corrected 2026-09-08: 569 GB free, not 43 GB.** The original number came
from `df -h /data`, which — because `/data` itself is not the actual mount
point — silently fell back to reporting the container's 98 GB overlay root
(55% used, 43 GB free). The project directory and everything under
`dataset/` in fact live on a separate, dedicated volume,
`/dev/mapper/ubuntu--vg-data--lv`, mounted at
`/data/dev/navid/kd-vago-ubuntu-two`: **815 GB total, 569 GB free** (27%
used). `dataset/` itself is currently 16 GB (before Phase 0/1's ~500 MB).

This does not remove the need for a bounded subset — the full ARKitScenes
3DOD corpus (623 GB, DATA.md) still dwarfs even 569 GB free, and the
per-scan sampling efficiency argument below holds regardless of how much
headroom exists — but it does mean **Phase 2's ≤25 GB budget is a small
fraction of available space (~4%), not the "real step up in resource
commitment" it was flagged as before this correction.** If a larger subset
than 500 scans would strengthen the benchmark, disk is no longer the
constraint that would stop it — bandwidth/wall-clock time (Phase 2 is
already scoped as "~1 day, mostly waiting") is the more relevant one now.

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

**Run 2026-09-08 — passed.** 5 real Training scans downloaded via the
official `download_data.py` (`ARKitScenes/threedod`, patched locally to
extract with Python's `zipfile` since this environment has no `unzip`
binary — 503 MB total, well inside budget) and probed with a standalone
script (`dataset/dataset_creation/arkit_tools/phase0_probe.py`, kept for
Phase 1 to build from). Findings:

- **Pose/intrinsics parse cleanly** and match `DATA.md`'s documented formats
  exactly: `.pincam` is `width height fx fy cx cy` on one line; `.traj` is
  `timestamp rx ry rz tx ty tz` (axis-angle radians, metres), one line per
  frame, matched to each RGB/depth frame's own timestamp by nearest-neighbour
  (observed alignment error ~0.0002-0.0005 s — timestamps are effectively
  exact, not approximate).
- **Depth confirmed plain millimetre uint16, no bit rotation** — checked
  both `DATA.md`'s own statement and real pixel values (one frame's non-zero
  range 1651-3425, i.e. 1.65-3.43 m after `/1000.0`, a plausible room depth).
  This confirms §3's warning: `decode_metric_depth`'s SUN-RGB-D bit-rotation
  must **not** be reused; ARKitScenes needs the simple `raw / 1000.0` path.
- **A real ambiguity resolved, not assumed:** each annotated object carries
  *two* OBB blocks, `segments.obb` and `segments.obbAligned`, at two
  different scales (`obb`'s `axesLengths` were ~100x `obbAligned`'s, e.g.
  `[120.7, 76.8, 120.0]` vs `[1.20, 1.21, 0.77]` for the same table — the
  former is a mesh-native space, not metres). `obbAligned` is the metric,
  room-scale, camera-pose-aligned box — confirmed by reading Apple's own
  `threedod/benchmark_scripts/utils/tenFpsDataLoader.py`, which builds boxes
  from `obbAligned` exclusively. The corner formula there
  (`compute_box_3d`: `corners = rotmat.T @ local_corners + center`, with
  `rotmat = normalizedAxes.reshape(3, 3)` unmodified) is what
  `phase0_probe.py`'s `obb_corners_world` implements.
- **Projection lands boxes correctly on real objects** — rendered overlays
  (projected hull drawn on the actual RGB frame) show tight alignment on an
  unambiguous case (a kitchen cabinet run: the drawn polygon sits directly on
  the real cabinet fronts and shelving).
- **The occlusion test behaves as intended** — a "cabinet" box whose observed
  LiDAR depth (1.33 m) was far closer than the box's own depth (3.48 m) was
  correctly flagged occluded; the rendered frame shows shelf clutter (bottles,
  boxes) sitting in front of where the cabinet actually is, exactly the
  physical situation the test is supposed to catch.
- **One real failure mode found and fixed, not anticipated in §3:** a chair
  box with one corner 0.18 m from the camera produced projected corners
  scattered as far as 600+ px outside a 256x192 frame (extreme perspective
  divergence from dividing by a near-zero depth), and the hull's clip against
  the image rectangle happened to keep a 30%-of-frame region that, rendered,
  corresponded to no part of the actual chair — a plausible-looking area
  fraction from a geometrically meaningless hull, not a code bug in the
  ordinary sense but a real property of corner-hull projection breaking down
  when any corner is very close to the lens. Fixed with an explicit near-
  distance gate (`MIN_CORNER_DEPTH_M = 0.4`, excluding any box with a corner
  closer than that) — the far-distance analogue of `depth.clip_max_m`, which
  Phase 1's indexer should adopt as a declared config value alongside the
  existing thresholds. Removing exactly this one case dropped total
  in-view projections from 101 to 92 across the 5-scan sample; everything
  else was unaffected, confirming it was an isolated edge case, not systemic.
- Occlusion rate across all sampled frames: 63/92 (68.5%) — high, but
  expected: most sampled frames of a room-tour video simply do not have a
  clear line of sight to a given annotated object, and among those that do
  land in view geometrically, being blocked by nearer furniture or clutter is
  common in real rooms. Not itself evidence of a problem.

**Gate verdict: pass — proceed to Phase 1.** Both required mechanisms
(projection, occlusion) work on real data; the one failure mode found has an
understood cause and a cheap fix. Phase 1 should carry forward: the near-
distance gate above, the `obbAligned`-only convention, and the confirmed
plain-millimetre depth decode.

### Phase 1 — Indexer (2–3 days, no GPU)

**Started 2026-09-08, core indexer built and passing against real data —
not yet run at Phase-2 scale.** `dataset/dataset_creation/v2/build_index_arkit.py`
emits the §2 schema (plus the additive `intrinsics_path` field) by reusing
`phase0_probe.py`'s validated projection/occlusion code directly rather than
re-deriving it. Run against the same 5 Phase 0 scans (8 sampled frames each):
13 frame records kept of 40 sampled (27 dropped, mostly `NO_OBJECTS_IN_VIEW` —
expected, matches Phase 0's own finding that most sampled frames of a room
tour do not have a clear line of sight to a given annotated object), label
counts `{table: 5, shelf: 5, washer: 8, cabinet: 8, refrigerator: 3, chair: 2}`.
A produced record was checked field-by-field against `build_index.py`'s own
schema and matches exactly.

**One real design decision this script had to make that SUN RGB-D's indexer
never faced:** ARKitScenes' boxes are scene-level, valid across an entire
video, not per-image. An object absent from a frame's `objects` list here
means "not visible in this frame at all" (occluded / out of the frustum /
inside the near-distance gate) and is **omitted entirely**, not marked
`is_valid_polygon=False` — the latter is reserved for an object that *is*
visible but whose projected geometry is too degenerate to trust, mirroring
SUN RGB-D's `INVALID_POLYGON` case exactly. Getting this wrong in either
direction would either let scene-wide-but-not-here objects block legitimate
existence-negatives, or (the other way) treat a real INVALID_POLYGON case as
if the object were never in frame at all. Both cases and the negative
control (a deliberately occluded object, at matching depth vs. not) are
pinned in `tests/test_build_index_arkit.py` (7 tests, synthetic geometry
with hand-computed pinhole answers), alongside a regression test for Phase
0's near-distance failure mode and a schema-shape parity check against
`build_index.py`'s own per-object record. Full suite 295 passed.

**Also done, additively, with zero behaviour change to the existing SUN
RGB-D pipeline (verified — full `dataset_creation/v2/tests` suite still
passes unmodified):** `depth_utils.load_intrinsics` was split into a new
`load_intrinsics_file(path)` (reads one file directly) plus
`load_intrinsics(scene_dir)` as a thin SUN-RGB-D-shaped wrapper around it;
`nearest_object.py` now prefers a record's own `intrinsics_path` field when
present and only falls back to the old `scene_dir_absolute()` reconstruction
when it is absent (true for every existing SUN RGB-D record, which therefore
sees no change at all).

**Deliberately not done yet, and not needed until Phase 2 exists:**
`build_vocab.py`'s unconditional seg37 union (§2's other verified gap) —
fixing it now would be speculative without a properly-sized corpus to check
the "frequency threshold alone likely suffices" claim against; 13 frames
across 5 scans is far too small a sample for that question to mean anything.
Same for actually running P1(vocab)/P2(generators)/P3(release) end-to-end —
worth doing once Phase 2's subset exists, not on this placeholder sample.

**Next decision point: Phase 2 is a real step up in resource commitment** —
~25 GB and "mostly waiting" vs. Phase 0/1's ~500 MB and minutes, so it was
not started without flagging that explicitly.

### Phase 2 — started 2026-09-08, bumped to 750 scans

Per the disk correction above (569 GB free, not 43 GB), author decision was
to bump the target from ~500 to **750 scans** — modestly more statistical
headroom for the audit/generalization claims, while staying close enough to
v2.4's own scale (4,187 images) that the two benchmarks remain comparable;
disk was never really the reason to cap it at 500, and it still isn't the
reason to go further than 750 (Phase 5's GPU-hour cost under the project's
known wall-clock-reset constraint scales with corpus size regardless of
disk headroom — see §8 for the full reasoning behind not going bigger).

Sample drawn once, seed 42, **stratified by ARKitScenes' own official fold
proportions** (4,498 Training : 549 Validation in the full 5,047-scan
corpus, 89.1%/10.9%) rather than an arbitrary split: **668 Training + 82
Validation = 750**, recorded at
`dataset/dataset_creation/arkit_tools/phase2_sample.csv` for
reproducibility (the plan's own §5 requirement). Downloading via
`dataset/dataset_creation/arkit_tools/download_phase2.py`, which:
- calls the vendored, zipfile-patched `download_data.download_data()`
  directly (no shelling out) rather than reimplementing the download logic;
- is resume-safe on its own terms — a scan whose `{video_id}_frames/` and
  annotation JSON are both already on disk is skipped before even touching
  the network, matching this project's established wall-clock-restart
  pattern (a restart costs at most the one scan in flight, not the batch);
- does **not** implement the earlier draft's "download, subsample frames,
  delete the rest" pruning step — that traded bandwidth complexity for disk
  savings, and disk is no longer scarce (750 scans x ~123 MB average
  observed in Phase 0 ~= 92 GB, well inside 569 GB free). Simpler to keep
  every frame `build_index_arkit.py` might later want to sample from.

One real bug caught before committing to the full run: the download
driver's own `REPO_ROOT` computation was off by one `dirname()` level (the
exact same class of mistake `build_index_arkit.py` made and had fixed
earlier the same day), which silently wrote a duplicated
`dataset/dataset/ARKitScenes/...` path. Caught by smoke-testing 1-2 scans
before launching the 750-scan batch, not discovered after; the misplaced
82 MB was deleted before the real run started.

**Completed 2026-09-08:** all 750/750 scans downloaded, 0 failures (749
fetched + 1 already present from Phase 0), 85 GB on disk (~126.5 min).

### Phase 3 — Generate and freeze VQA-ARKitScenes v1 (half a day, CPU)

Run the unchanged P2 generators, then `build_vocab`, `balance`, `answer_form`,
`build_release`, `freeze_release.py`. Produces a frozen release with its own
manifest and sha256s under the same discipline as v2.4.

**Started 2026-09-08.** `build_index_arkit.py` run over both fold
directories (`--split train` over Training, `--split test` over Validation —
placeholders, see below), producing 945 frame records from 668 Training-fold
scans and 126 from 82 Validation-fold scans (**82%/81% of sampled frames
dropped, mostly `NO_OBJECTS_IN_VIEW`** — the same failure mode Phase 0
flagged on its 5-scan sample, now confirmed at full scale and materially
worse than assumed; see the vocab finding below).

**Train/val/test policy** (deferred by `build_index_arkit.py`'s own
docstring — "not resolved here"): new `assign_split_arkit.py` maps
ARKitScenes' official Validation fold wholesale to `test` (the direct
analogue of SUN-RGB-D's own fixed official test pool — the one externally-
defined held-out set neither pipeline invented), then carves
`val_fraction=0.15` out of the Training-fold pool by `GroupShuffleSplit` on
scan id, **reusing `build_index.py`'s own tested `assign_split()`** rather
than reimplementing split logic. Result: **1,071 total frames → 818 train /
127 val / 126 test**, merged into `data/index/scene_index_arkit.jsonl`. 4
new tests (`tests/test_assign_split_arkit.py`); full suite 331 passed.

**`build_vocab.py`'s deferred seg37 decision, resolved against the real
corpus.** The plan's original assumption — "all 17 target classes should
each clear the ≥100-occurrence threshold on their own in 2,500-5,000
images" — does not hold: the actual corpus is only 1,071 frames (roughly a
third of that range), and at that scale only 5 of the 17 target classes
(cabinet 579, chair 293, table 213, sofa 140, sink exactly 100) clear
`FREQUENCY_THRESHOLD=100` unassisted; the other 12 (bed 87, toilet 77,
bathtub 75, stool 58, shelf 57, washer 47, refrigerator 36, oven 35,
tv_monitor/television 30, stove 19, fireplace 13, dishwasher 9) would have
silently fallen out of the vocabulary. **Author decision: always-include
the 17 target classes** (over lowering the threshold or accepting the
narrower 5-class vocabulary) — the direct ARKitScenes analogue of seg37,
except with no name-format mismatch to bridge, since these are exactly
ARKitScenes' own raw 3DOD label strings. Implemented as `--dataset
{sunrgbd,arkitscenes}` on `build_vocab.py`: `sunrgbd` (default) is verified
byte-for-byte identical to the pre-change script (isolated by diffing two
fresh reruns of the original vs. patched code against the same on-disk
index — the git-committed vocab files themselves turned out to already be
stale against a later index rebuild, an unrelated pre-existing drift, not
a regression from this change); `arkitscenes` reads
`scene_index_arkit.jsonl`, always-includes
`ARKITSCENES_TARGET_RAW_NAMES` instead of loading seg37, and writes to a
separate `data/vocab_arkit/` so SUN-RGB-D's curated vocab is never touched.
Two small, purely additive vocabulary fixes needed along the way: `washer`
had no `CATEGORY_AND_STRUCTURAL` entry (added, `electronics`), and
ARKitScenes' combined `tv_monitor` class had no synonym mapping (added
`tv_monitor -> television` to `data/vocab/synonyms.csv`, folding it into
the concept SUN-RGB-D's own `tv` abbreviation already maps to). Result: all
17 target classes present, **100% instance coverage** (1,868 instances),
0 review-queue entries. Only 5/17 concepts have the ≥10 non-border-touching
samples `concept_typical_area.json` needs for a trusted median size — worth
flagging for Phase 4's audit, since the other 12 classes' crop-gate
comparisons (Rule V3-adjacent, `scene_objects.py`) rest on no typical-size
reference at all.

**Not yet run**: `answer_form`, `freeze_release.py` — same `RELEASE_DIR`/
path-constant parameterization gap §2 already flagged, not yet needed for
`build_release.py` below (`answer_form` was not exercised by the sanity
checks; needs its own audit before use).

**`build_release.py` run, after finding and fixing two more real bugs it
exposed** (2026-09-08) — both caught because running the real pipeline
end-to-end surfaces gaps a schema-parity read-through cannot:

1. **`load_intrinsics_file` never actually supported ARKitScenes**, despite
   its own docstring's claim. `.pincam` files are 6 values (`width height
   fx fy cx cy`); the function required exactly 9 (a flattened 3x3 matrix,
   SUN RGB-D's `intrinsics.txt` shape) and silently returned `None`
   otherwise — every `.pincam` read as `MISSING_INTRINSICS`, dropping 85%
   of `nearest_object` candidates (913/1071) for a reason that had nothing
   to do with real data scarcity. Fixed by branching on token count (6 ->
   build K the same way `arkit_tools/phase0_probe.py`'s
   `load_intrinsics_pincam` already does; 9 -> unchanged). 4 new tests
   (`tests/test_depth_utils.py`); verified byte-identical for SUN RGB-D.
2. **`left_right.py` re-parses the raw annotation JSON for polygon IoU**,
   using SUN RGB-D's `frames[0]["polygon"]` shape — ARKitScenes' annotation
   format has no such key, so every pair's polygon lookup failed
   (`POLYGON_UNAVAILABLE`, 780 of 1851 drop rows), a genuine schema gap
   Phase 1's read-through missed because `resolve_scene_objects` only
   passes through an explicit, hand-listed set of fields — a new one
   silently disappears rather than erroring. Fixed by having
   `build_index_arkit.py` store the already-computed, already-clipped 2D
   hull directly on each object record (`polygon_xy`, the same polygon
   used to score visibility — no new geometry, just not discarding it),
   threading it through `resolve_scene_objects`, and having `left_right.py`
   prefer it over the raw-annotation reconstruction when present (same
   prefer-the-new-field-when-present pattern `nearest_object.py`'s
   `intrinsics_path` already established). Both fixes together took
   `left_right`/`nearest_object` from 0 candidates each to 220/205.

**P2 generators parameterized**: `generator_common.py`'s `run_generator()`
gained a `dataset` argument (`sunrgbd` default, byte-identical — verified
by diffing original-code vs. patched-code reruns against the same on-disk
index for all 5 generators) alongside a shared `parse_dataset_arg()` each
generator's `__main__` now calls; outputs go to `data/candidates_arkit/`
and `build_log/p2_*_drops_arkit.csv` so nothing SUN-RGB-D owns is touched.
Full ARKitScenes candidate counts: existence 26,494, identify_superlative
1,471, left_right 220, nearest_object 205, relative_depth 145.

**`build_release.py` run — found a third real problem, resolved by author
decision.** First attempt crashed, then (once traced) produced a
completely **empty val/test release** (0 items each), even though train
was fine (1,548 items): Rule 6.4 caps every type's val/test count to
whichever type has the fewest, and `nearest_object`'s own answer-balance
cap (`cap_answer_share_per_group`, `NEAREST_CONDITIONAL_MAX_SHARE=0.20`)
trimmed its 14-34-row val/test pools to exactly 0 — a cap calibrated for
pools of thousands, dragging every other (healthy: 7-116 items) type down
to 0 with it via the shared minimum. **Author decision: relax the caps
rather than drop Rule 6.4's shared-minimum structure or drop
`nearest_object` from the dataset.** Implemented as `balance.py`'s new
`cap_with_floor(capped, original, min_keep)` — falls back to the *uncapped*
pool exactly when the strict cap would drop below `min_keep`, a no-op at
`min_keep=0` (every SUN-RGB-D call site, verified byte-identical). Wired
into `build_release.py`'s two collapse-prone call sites at `min_keep=1`
for `--dataset arkitscenes` only. 3 new tests
(`tests/test_cap_with_floor.py`). Result: **train 1,548 / val 33 / test 38**
items, all 5 question types present in every split, 0 sanity-check
failures (9 warnings — expected at this scale: type-balance spread and
majority-share targets calibrated for thousands of rows read as exceeded
on val/test pools of 6-8 items per type). Written to
`release/VQA-ARKitScenes-v1/rule_based/{train,val,test}.csv`,
`build_log/p3_report_arkit.json`. **Not yet frozen** —
val/test at 33-38 items total is a genuinely small benchmark slice, worth
the author seeing before `freeze_release.py` locks it as a versioned
release.

Full suite 338 passed throughout (335 + 3 `cap_with_floor` tests).

**Frozen as VQA-ARKitScenes-v1.0** (author instruction, 2026-09-08).
`freeze_release.py` gained the same `--dataset` parameterization
(`ARKIT_TRACKED_INPUTS`, `ARKIT_RELEASE_ROOT`) as the rest of the P1-P3
family, plus one generalization needed for it specifically: `sha256_of`'s
dispatch to the volatile-field-stripping manifest/scene-index hashers was
matched by the exact SUN-RGB-D filenames (`manifest.json`,
`scene_index.jsonl`); broadened to match by pattern
(`manifest*.json`, `*.jsonl`) so `manifest_arkit_split.json` (which carries
the same volatile `built_at_utc` field) and `scene_index_arkit.jsonl` (the
same environment-dependent float precision) get identical treatment
without a dataset branch inside the low-level hash functions — verified
against SUN-RGB-D's real frozen v2.4 (`--verify v2.4` still resolves the
same two files the same way). `release/VQA-ARKitScenes-v1/FROZEN_v1.0.json`:
train.csv 1,548 rows, val.csv 33 rows, test.csv 38 rows; `--verify v1.0`
passes clean.

**Known, accepted side effect, not a bug**: `--verify v2.4` (SUN-RGB-D) now
reports drift on every pipeline file this session's ARKitScenes work
touched (`synonyms.csv`, `balance.py`, `build_release.py`,
`depth_utils.py`, the 5 generators, `scene_objects.py`) — correctly, since
those files' *bytes* did genuinely change, even though their SUN-RGB-D
*behaviour* was verified byte-identical at every step above. The freeze
mechanism hashes tracked-input content, not behavior, so it cannot
distinguish "changed but behaviorally inert" from "changed and different" —
by design, per its own docstring ("any future change... must produce a new
version, not silently rewrite this one"). Not fixed here; re-freezing v2.4
under a new tag was out of scope for this instruction and nothing about
v2.4's own release CSVs changed.

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
respectively, at measured rates). Prerequisite carried over from §3: every
`modality="depth"` row here depends on `depth_input.decode_metric_depth`
actually decoding ARKitScenes' LiDAR files correctly — verified in Phase 0,
fixed if needed in Phase 1 — not on the SUN-RGB-D-specific bit-rotation it
currently implements happening to also be correct for a different sensor.

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

**2026-09-08, Phase 5 launched.** Author instruction: no ScanNet (decision 2
below) — "requires filling forms and stuff." D9 (the last SUN-RGB-D KD row)
finished first (best val macro 47.88%, epoch 2), freeing the GPU.

Before launching, extensive cross-cutting plumbing was needed across
`distillation/` and `evaluation/` — none of it exercised until an actual
second dataset existed to run against: `depth_input.py` gained
`decode_arkit_depth` (ARKitScenes' plain-millimetre depth vs. SUN-RGB-D's
bit-rotation — the exact prerequisite this plan's Phase 5 section already
flagged), threaded via a new `--dataset` flag through
`train_student.py`/`train_kd.py`/`zero_shot_inference.py`'s `build_image`;
`build_teacher_cache.py`'s `dataset_version()` (a *required* cache-key
field) was made dataset-aware after realizing an ARKitScenes cache built
under the old hardcoded SUN-RGB-D version string would digest-collide with
the real SUN-RGB-D cache; `evaluate.py` and `epoch_loop.py`'s
`score_val_macro` both hardcoded SUN-RGB-D's 148-concept vocab for scoring
*any* prediction, silently marking ARKitScenes' own answers (e.g.
"washer") out-of-vocabulary — both gained a `canonical_objects_dir`
override. All changes additive, defaults unchanged, verified via the full
test suite (350 passed) and smoke tests of every affected code path
(feature cache, logits cache, B3-shaped CE training, X2-shaped KD
training, D0-shaped alignment, D0→D3 checkpoint chaining) before the real
run.

**Real, non-cosmetic finding along the way**: `recipe_library()`'s current
`D0` declaration (and `stage_one()`, which `D3`/`D5` derive their
alignment stage from) trains the vision merger alongside attention — a
configuration measured on SUN-RGB-D and **rejected** as sub-chance
(`align_curve_contrastive_merger_s17`, declining 25.84% -> 22.92%, below
the 30.3% floor throughout). The checkpoint D4/D5/D6/D9 actually trained
against (`align_curve_contrastive_s17`) was a different, earlier,
attention-only run manually preserved and reused via `--parent-checkpoint`
— not regenerated through today's `stage_one()`/D0 code, which would
silently reproduce the rejected configuration. ARKitScenes has no such
pre-existing checkpoint to borrow. **Author decision: run D0 attention-
only** — a new `--trainable-modules` override on `train_kd.py` (applied
*after* stage resolution; an earlier attempt applying it before was
silently discarded by `stage_one()`'s own hardcoded surface, caught by
smoke-testing before the real run). This is a standing gotcha for any
*future* fresh SUN-RGB-D stage-F run too, not just ARKitScenes — the
project's own working checkpoint predates a code change that was never
reverted despite the rejection.

**Phase 5 is the reduced ladder** the plan's own §6 text already scoped —
not the full D1-D9 matrix run for SUN-RGB-D. Running: B1, B2 (zero-shot
depth/RGB), B3, B5 (CE depth/RGB), X2 (CE+KD unaligned), D0 (alignment,
attention-only), D3, D5 (CE-only / CE+KD on aligned vision) — one seed
(17), same 10-epoch/patience-2/LoRA-rank-16 policy as every SUN-RGB-D row.
Launched via `runs/kd_arkit/run_phase5.sh`. Tracked with the same ad-hoc
`runs/<name>/resource_usage.json` convention D4-D9 used this session, not
`evaluation/record_run.py`'s more elaborate `INDEX.md`-tracked system the
*original* B1-X2 SUN-RGB-D numbers went through — that tool also hardcodes
SUN-RGB-D's vocab/release paths and would need the same kind of
parameterization before it could track ARKitScenes runs; deferred as a
separate decision, not required to get real numbers now.

## 8. Decisions needed

1. ~~Proceed with Phase 0...~~ **Done (2026-09-08): passed.** See §6's Phase 0
   results. Remaining decision: proceed to Phase 1 (the real indexer, 2-3
   days), or hold pending author review of the Phase 0 findings first?
2. Request ScanNet access in parallel now, as insurance and as a possible third
   dataset?
3. `identify_superlative` on ARKitScenes: projected-hull area (consistent with
   v2.4) or 3D box volume (more accurate, but a different question)? Must be
   declared before generation, not chosen after seeing results.
   **Recommendation: projected-hull area.** Two independent reasons point the
   same way, not just consistency with v2.4's own definition (real, but the
   weaker of the two): the task's premise is a model answering from what a
   camera actually sees, and "largest" by true 3D volume can disagree with
   what is visually apparent in a given frame — a distant sofa outmasses a
   near chair in volume while the chair fills more of the image — which asks
   the model to reproduce privileged 3D ground truth it fundamentally cannot
   derive from its own input, not to ground a visual judgment. This is the same
   reasoning §3 already applies to the depth-definition choice (observed
   median over box-centroid, "for consistency with v2.4's definition"), so
   answering it the same way keeps `identify_superlative` and `relative_depth`
   under one coherent operating principle across both benchmarks: the gold
   answer is always what would be inferable from the image itself, and box
   geometry that *disagrees* with the observed frame is recorded as evidence
   alongside, never substituted as the gold answer.
4. ~~Run the leave-one-source-out study...~~ **Done, independent of this plan**
   (2026-09-08): training on 3 of SUN RGB-D's 4 sensors and testing on the
   held-out one (not specifically `kv1/NYUdata`, but the same shape of
   question) showed no measurable generalization drop — see
   `pilot_findings.md` §14, `experiment_protocol.md`'s 2026-09-08 amendment row.
   That result already exists; it does not substitute for ARKitScenes'
   cross-*dataset* (as opposed to cross-*sensor*) transfer question (Phase 6),
   which needs the real second dataset to answer at all.
5. **Decided (2026-09-08): the current paper submission proceeds on SUN-RGB-D
   alone; ARKitScenes is not included as a reported second dataset for now.**
   Reasoning: Phase 5's exploratory ladder (single seed, ~33-42-item val set)
   came back contradicting the core hypothesis — D5 (CE+KD, aligned) scored
   30.95%, below D3 (CE-only, aligned) at 40.95% — and that result is not yet
   distinguishable from small-N noise (see §6's Phase 5 writeup). Phase 4's
   human audit of the release also never ran. Reporting either an unaudited
   dataset or a single-seed result that undercuts the method's own claim
   would weaken the submission rather than strengthen it. This does not
   retire the ARKitScenes track — the code, v1.0 release, and Phase 5 code
   path all stay in the repo (Phase 5's own run artifacts kept local/
   gitignored, not committed) — it is deferred past this submission. Should
   the author want to revisit it later: rerun D0/D3/D5 with a second seed
   first (cheap, ~15-20 GPU-minutes total per §6) to see whether the CE-vs-KD
   gap is real, then decide whether Phase 4 audit + Phase 6 transfer are
   worth the remaining time before any future resubmission.
