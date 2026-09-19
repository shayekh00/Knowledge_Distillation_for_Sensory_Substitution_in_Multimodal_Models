# M³FD Thermal-VQA execution runbook

This runbook runs the implementation without modifying VQA-SUNRGBD-v2.4. All
paths below are relative to the repository root. Do not run release or freeze
steps until the Phase-0 manual evidence has passed.

## Evidence required before construction

Place the licensed, extracted source below `dataset/` (for example,
`dataset/M3FD`). Determine from the source documentation and probe whether XML
boxes are in `rgb` or `thermal` pixels, and determine a capture-session regular
expression with a capture group. A frame number is not an acceptable group.

Run the read-only probe, recording the real source URL and terms:

```bash
python dataset/dataset_creation/v2/probe_m3fd.py \
  --source-root dataset/M3FD --rgb-dir Visible --thermal-dir Infrared \
  --annotation-dir Annotation --box-coordinate-frame rgb \
  --source-url 'SOURCE_URL' --license-terms 'TERMS_URL_OR_PATH' \
  --out-dir build_log/m3fd/probe
```

Review its 100 overlays. Complete `registration_review.jsonl` with a boolean
`registration_valid` on every sampled row. Construction is blocked unless at
least 95% are valid and capture grouping is defensible.

**Review coverage vs `--require-registration-review`.** These two are not the
same gate, and the build command below will silently enforce the stricter one.
`build_index_m3fd.py --require-registration-review` admits *only* frames whose
`image_id` is present in the review file with `registration_valid: true`;
everything else is dropped as `REGISTRATION_UNREVIEWED`. The probe samples 100
frames, so running the two commands exactly as written yields a 100-frame
index and 4,100 dropped pairs, and the release then fails its
200-rows-per-type floor. Pick one deliberately:

* keep `--require-registration-review` and extend the review file to cover
  every frame you intend to index, or
* treat the 100-frame review as the sampled 95% gate it is described as, omit
  `--require-registration-review`, and pass `--registration-review` alone so a
  frame reviewed and *rejected* is still excluded while unreviewed frames are
  indexed.

Check `build_log/m3fd/index_drops.csv` for `REGISTRATION_UNREVIEWED` after the
build either way.

## Build commands

```bash
python dataset/dataset_creation/v2/build_index_m3fd.py \
  --source-root dataset/M3FD --rgb-dir Visible --thermal-dir Infrared \
  --annotation-dir Annotation --box-coordinate-frame rgb \
  --capture-group-regex '(?P<group>YOUR_SESSION_PATTERN)' \
  --registration-review build_log/m3fd/probe/registration_review.jsonl \
  --require-registration-review --annotation-complete

python dataset/dataset_creation/v2/near_duplicates_m3fd.py \
  --index data/index/scene_index_m3fd.jsonl --dataset-root dataset \
  --out data/index/near_duplicates_m3fd.jsonl

python dataset/dataset_creation/v2/assign_split_m3fd.py \
  --index data/index/scene_index_m3fd.jsonl \
  --near-duplicate-groups data/index/near_duplicates_m3fd.jsonl

python dataset/dataset_creation/v2/m3fd_candidates.py
python dataset/dataset_creation/v2/build_release_m3fd.py
```

`--annotation-complete` is a source-evidence assertion. Omit it unless a
review confirms the annotations are complete enough for existence and count;
those two types will otherwise be excluded at candidate generation.

## Evaluation, audit, and freeze

Evaluate M³FD predictions with its own vocabulary and all four answer spaces:

```bash
python evaluate.py --dataset m3fd \
  --release-dir release/VQA-M3FD-Thermal-v1/rule_based \
  --canonical-objects-dir data/vocab_m3fd --synonyms-dir data/vocab_m3fd \
  --predictions PREDICTIONS.csv
```

Before adding `DATASHEET.md` and freezing, audit 150 stratified test examples
per retained type against both source modalities and the annotation overlay.
The audit app serves that review directly — `m3fd` is one of its named sources
(`tools/audit_app/sources.py`), so no path juggling is needed:

```bash
# 150 per retained type, from the M3FD release. Unstratified within type:
# M3FD release rows carry sequence_id, not the `sensor` column the SUN-RGB-D
# sampler stratifies on, and every frame is the same thermal sensor anyway.
python -m tools.audit_app.sampling --source m3fd

AUDIT_SOURCE=m3fd python -m uvicorn tools.audit_app.main:app --port 8002
```

The reviewer sees the **thermal** frame by default, with the question's
evidence boxes outlined; `r` (or the modality buttons) flips to the registered
**RGB** frame with the same boxes rescaled into it, which is the
both-modalities check this step calls for. Thermal is the frame gold has to be
true of — it is the only image the student ever receives — and RGB is the
cross-check, not the subject. Overlays come from the evidence each generator
recorded: every box for `existence`, every box of the counted class for
`count`, both compared boxes for `left_right`, and winner *and* runner-up for
`identify_superlative`, so its 1.20x area margin is actually checkable. A
question whose gold is a negative `existence` "no" has no evidence box, and
the overlay falls back to showing every annotated object so the absence can be
confirmed.

Then render the report and read the per-type acceptance rule off it:

```bash
AUDIT_SOURCE=m3fd python -m tools.audit_app.report
```

Remove any type that does not meet the approved audit threshold; then write the
datasheet and run:

```bash
python dataset/dataset_creation/v2/freeze_m3fd.py
python dataset/dataset_creation/v2/freeze_m3fd.py --verify
```

The thermal student loader is `dataset.dataloader.M3FDThermalDataset`; it
opens only `thermal_path` and returns no RGB or annotation metadata.
