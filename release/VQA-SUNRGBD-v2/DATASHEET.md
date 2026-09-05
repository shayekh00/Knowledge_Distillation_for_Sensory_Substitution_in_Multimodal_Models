# Datasheet — VQA-SUNRGBD-v2

Following Gebru et al., *Datasheets for Datasets* (2021). Version **v2.4**,
frozen 2026-09-05. Every number here is measured from the frozen release; the
build rules are specified in `docs/DATASET_CREATION_PLAN.md`.

## Motivation

**Why was the dataset created?** To evaluate whether a multimodal model has
learned spatial structure that is *grounded in depth*, rather than recovered
from language priors — the setting of a knowledge-distillation study in which
a depth branch is distilled into an RGB-only student. Its predecessor
(VQA-SUNRGBD-v1) could not support that claim: a majority-class baseline was
competitive with trained models, no answer was derived from measured depth,
and object names had been corrupted by a neural spell-corrector.

**Who created it?** Shayekh Mohiuddin Ahmed Navid. Built from SUN RGB-D
(Song et al., CVPR 2015).

## Composition

**What do instances represent?** One question about one SUN RGB-D frame, with
a single short gold answer. Instances carry the RGB and depth paths, sensor,
scene type, question type, template id, answer space, and a JSON `evidence`
field recording the annotated objects the answer was computed from.

**How many instances?**

| Split | Items | Images |
|---|---:|---:|
| train | 15,278 | 4,187 |
| val | 1,720 | 667 |
| test | 12,463 | 4,703 |

Five question types, each 2,470–2,558 items in test:
`existence`, `identify_superlative`, `left_right`, `nearest_object`,
`relative_depth`. Three of the five (`identify_superlative`,
`relative_depth`, `nearest_object`) are answerable from measured depth alone.

**Does the dataset contain the images?** No. Rows reference SUN RGB-D frames
by relative path; the imagery must be obtained from the original distribution.

**Is any information missing?** Yes, by construction. Only objects in the
151-concept canonical vocabulary can be a gold answer; that vocabulary covers
79.2% of annotated object instances (81.0% of answer-eligible ones). Out-of-
vocabulary objects still count as present for existence negatives, so they are
never treated as absent, they simply cannot be asked about.

**Are there errors or noise?** Yes, measured rather than assumed — see
*Gold verification* below and `stats/report.md`.

## Collection process

Answers are computed deterministically from SUN RGB-D's own 2-D polygons, 3-D
boxes, `depth_bfx` depth maps, and `scene.txt`, using a single global seed
(42). No model produces any gold label. Depth-derived answers use the official
SUN RGB-D depth decoding; an object's depth is the median of valid masked
pixels and is used only when ≥30% of those pixels are non-zero. Polygons are
clipped to the RGB frame before any geometry is measured.

**Sampling.** Every scene that survives the gates contributes up to 4 questions
of distinct types in val/test and up to 8 in train. Each type applies margin
gates so that the answer is unambiguous (for example, `identify_superlative`
requires the second-closest object to be at least 1.2× the closest object's
depth).

**Splits.** Test is the official SUN RGB-D test split, so results are
comparable with other SUN RGB-D papers. Train/val is a seeded, sequence-
grouped split of the official train pool. Because SUN3D sequences are
keyframes from continuous video, images sharing a `<building>/<room>` group are
kept together, and 302 train/val images sharing a sequence with an official
test frame were dropped. Verified directly: zero image overlap and zero
sequence-group overlap between any two splits.

## Preprocessing and labelling

Object names are canonicalised deterministically — lowercase, strip trailing
digits, singularise with `inflect`, apply a hand-written synonym table, then
look up the curated 151-concept vocabulary. No spell-correction model is used
anywhere; v1's neural corrector is what produced `red → bed` and `hair → chair`.
Structural surfaces (`wall`, `floor`, `ceiling`, frames) are never gold
answers, though they may serve as reference objects.

Every rejected candidate is logged with a reason code: 42,939 drops in
`stats/drops.csv`.

## Balance and shortcut controls

`existence` and `left_right` are exactly 50/50; `existence` is additionally
paired yes/no per concept in every split. Open-vocabulary types are capped at
8% per answer, and `nearest_object` at 20% per answer within each anchor
concept. A TF-IDF logistic-regression **question-only** baseline is fit on
train and evaluated on val for every type; the build fails if it beats that
split's majority baseline by more than 5 percentage points. Measured excesses
for v2.4: existence +2.6, identify_superlative +0.6, relative_depth −7.0,
nearest_object +4.7, left_right +1.2 percentage points.

This is a guardrail, not a proof: it shows no *linear TF-IDF* shortcut of that
size exists, not that no shortcut exists.

## Gold verification

A stratified sample of 150 test items per type (750 total) was checked by one
reviewer against the RGB image and polygon overlay. Acceptance requires ≥95%
gold accuracy and ≤3% ambiguous.

| Type | Gold accuracy | Ambiguous | Accepted |
|---|---:|---:|---|
| existence | 82.0% | 0.0% | **no** |
| identify_superlative | 97.3% | 2.7% | yes |
| left_right | 95.3% | 2.0% | yes |
| nearest_object | 100.0% | 0.0% | yes |
| relative_depth | 100.0% | 0.0% | yes |

This is **single-reviewer gold verification**. It measures acceptance of the
released labels; it is *not* independent annotation and no inter-rater
agreement or Cohen's κ is claimed. Two types (`existence`, `left_right`) were
reviewed item by item; the other three were recorded as a type-level judgment
with `tools/audit_app/bulk_verdict.py` and are labelled as such in the
response log.

No dataset-level pooled accuracy is reported. Averaging five types built by
different generators hides exactly the per-type failure this protocol exists
to surface.

## Known limitations

* **D17 — `existence` negatives (unfixed).** Absence is tested on the
  canonical name alone, so a scene annotated `desk`, `counter`, or
  `coffeetable` can carry a gold `no` for `table`. Measured at 30.6% error on
  negative items in the audit sample; an estimated **902 of 1,650** negative
  `table` items release-wide sit on such a scene (538 test / 360 train / 4
  val). This is a lower bound — only the `table` family was quantified.
* **D16 — out-of-frame referents (unfixed).** Some items ask about an object
  that is mostly outside the frame. A filter was built and measured but could
  not be shown to help, and was therefore not adopted rather than applied on
  faith.
* **Validation split is small** (1,720 items, 340–352 per type), a consequence
  of strict split-preserving construction and per-concept balancing.
* **Vocabulary is a committed input, not a rebuild artefact.** Re-deriving it
  from scratch yields 148 concepts rather than the shipped 151.

## Uses

Intended for evaluating spatial and depth-grounded visual question answering,
and for training. Results should be reported with `evaluate.py`, which fixes
greedy decoding, the shared canonicaliser, macro accuracy over types as the
headline, and the random / majority / question-only baselines in the same
table.

**Not appropriate for:** claims about counting (the `count` type was retired
after its audit measured 40% gold accuracy — SUN RGB-D's polygon annotations
are not exhaustive per instance); claims about color (removed, since v1's
color labels were another model's guesses); and, until D17 is fixed, claims
resting on `existence` negatives alone.

## Distribution and maintenance

Distributed under CC BY-SA 4.0, inherited from SUN RGB-D — see `LICENSE`. Each
release is frozen with per-file and per-input SHA-256 in `FROZEN_<version>.json`
and verified with `freeze_release.py --verify`. Releases are immutable: a
defect is fixed by rebuilding forward and cutting a new version, never by
editing a frozen CSV. Prior audits are retained under `audit/*_archive/`.
