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
Remove any type that does not meet the approved audit threshold; then write the
datasheet and run:

```bash
python dataset/dataset_creation/v2/freeze_m3fd.py
python dataset/dataset_creation/v2/freeze_m3fd.py --verify
```

The thermal student loader is `dataset.dataloader.M3FDThermalDataset`; it
opens only `thermal_path` and returns no RGB or annotation metadata.
