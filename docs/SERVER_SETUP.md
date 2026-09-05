# Reproducing this project on another machine

How to get from a bare server to a working copy of this repository: the frozen
VQA-SUNRGBD-v2 dataset, the build pipeline, the audit tool, and the
distillation training runs.

Read this once before starting. The short version: **almost everything is in
git, and the two things that are not — the SUN RGB-D imagery and the toolbox —
are public downloads.** You do not need to copy 8 GB off this laptop.

---

## 1. What travels how

| What | Size | In git? | How to get it |
|---|---:|---|---|
| Code, tests, docs | small | ✅ | `git clone` |
| `release/VQA-SUNRGBD-v2/` — frozen CSVs, manifests, datasheet, stats | 24 MB | ✅ | `git clone` |
| `data/{vocab,splits,templates,config.yaml}` — curated inputs | 3 MB | ✅ | `git clone` |
| `audit/` — samples, response logs, archived audits | 7 MB | ✅ | `git clone` |
| `build_log/` — drop logs, P3 report | 5 MB | ✅ | `git clone` |
| `dataset/SUNRGBD/` — RGB + depth imagery | **7.3 GB** | ❌ gitignored | public download (§3) |
| `dataset/SUNRGBDtoolbox/` — `SUNRGBDMeta.mat`, `allsplit.mat` | **657 MB** | ❌ gitignored | public download (§3) |
| `data/index/` — P0 scene index | 55 MB | ❌ gitignored | regenerate (§5) |
| `data/candidates/` — P2 raw pools | 341 MB | ❌ gitignored | regenerate (§5) |
| `dataset/SUNRGBD/csv_data/*.csv` — v1-schema training CSVs | 8 MB | ❌ gitignored | regenerate (§4, one command) |
| `.env` — API keys and data roots | tiny | ❌ gitignored | recreate by hand (§2) |
| Model checkpoints (`checkpoints/`) | large | ❌ | copy manually, or retrain |

Two consequences worth internalising:

* `dataset/SUNRGBD/*` is gitignored **wholesale**, which sweeps up
  `csv_data/*.csv` — the CSVs the existing dataloaders read. They are not lost;
  they are a projection of the frozen release and are regenerated with one
  command (§4). Do not hand-copy them.
* `data/index/` and `data/candidates/` are gitignored because they are
  regenerable and large. You only need them if you intend to **rebuild** the
  dataset (§5). To train or evaluate on the released data, skip both.

---

## 2. Environment

Python 3.9, conda. The pinned `h5py==3.10.0` build does not work on a bare
system Python here, so use the environment rather than `pip install --user`.

```bash
git clone git@github.com:shayekh00/Knowledge_Distillation_for_Sensory_Substitution_in_Multimodal_Models.git
cd Knowledge_Distillation_for_Sensory_Substitution_in_Multimodal_Models
git checkout revised-paper-submission

conda create -n kd_env python=3.9 -y
conda activate kd_env
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.2.1` with CUDA 12.1 wheels. On a GPU server
with a different CUDA version, install torch first from the correct index and
then the rest:

```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

It also installs `transformers` from a pinned git commit and
`LLaVA-NeXT` from GitHub, so the server needs outbound access to github.com.
`flash-attn` is commented out; enable it only if the GPU supports it.

### `.env`

Not in git. Create it at the repository root with these four keys:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ROOT_DATA_DIR=/absolute/path/to/repo/dataset
MAIN_ROOT_DATA_DIR=/absolute/path/to/large/scratch/volume
EOF
```

* `ROOT_DATA_DIR` points at this repo's `dataset/` directory. The dataloaders'
  `remove_substring_from_path` depends on that, so a wrong value produces
  missing-image errors rather than a clear failure.
* `MAIN_ROOT_DATA_DIR` is where `checkpoints/` is written. Put it on a volume
  with room for 7B checkpoints, not in the repo.
* `DEEPSEEK_API_KEY` is needed only for the audit model-triage pass
  (§6) and the not-yet-built LLM-authored set.
* **The `OPENAI_API_KEY` that was in the old `.env` was found out of credits
  and is stored in plaintext. Rotate it rather than copying it across.** Never
  commit this file.

---

## 3. SUN RGB-D data

Both downloads come from <https://rgbd.cs.princeton.edu/>. Neither is in git,
and neither should be copied from this laptop — a fresh download is faster and
verifiable.

```bash
cd dataset
# RGB + depth frames -> dataset/SUNRGBD/{kv1,kv2,xtion,realsense}   (7.3 GB)
# toolbox            -> dataset/SUNRGBDtoolbox/                      (657 MB)
```

Expected layout after unpacking:

```
dataset/SUNRGBD/{kv1,kv2,xtion,realsense}/...
dataset/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat
dataset/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat
```

Those two `.mat` paths are read directly by
`dataset/dataset_creation/v2/build_index.py`; if either is missing, P0 fails
immediately rather than silently.

Sanity check: 10,335 frames total (959 MB kv1, 2.8 GB kv2, 1.5 GB xtion,
2.0 GB realsense).

---

## 4. Path A — use the frozen release (recommended)

This is what you want for training, evaluation, or writing the paper. It needs
the imagery from §3 but **no rebuild**, so it takes minutes.

```bash
conda activate kd_env

# 1. confirm the release on disk matches its manifest
python dataset/dataset_creation/v2/freeze_release.py --verify v2.4
# -> "v2.4: verified, no drift."

# 2. regenerate the v1-schema CSVs the dataloaders read (gitignored)
python dataset/dataset_creation/v2/export_v1_schema.py
# -> dataset/SUNRGBD/csv_data/{train,val,test}_dataset.csv

# 3. run the test suites
python -m pytest tests/ dataset/dataset_creation/v2/tests tools/audit_app/tests -q
# -> 79 passed

# 4. reproduce the published baselines
python evaluate.py --baselines-only
# -> macro accuracy 30.2% random / 33.1% majority / 34.9% question-only
```

If all four succeed, the server is reproducing this machine. Step 4 is the
strongest single check: it exercises the release CSVs, the vocabulary, and the
shared canonicaliser together, and its numbers are in
`release/VQA-SUNRGBD-v2/stats/baselines.md` to compare against.

The release is 15,278 train / 1,720 val / 12,463 test items across five
question types. `release/VQA-SUNRGBD-v2/DATASHEET.md` describes it, including
known defects.

---

## 5. Path B — rebuild the dataset from raw SUN RGB-D

Only if you are changing the pipeline. Everything is CPU-bound and takes
minutes, not hours. Run the phases **in order**, and only from the phase you
actually changed onward.

```bash
conda activate kd_env

# P0 — scene index (writes data/index/, gitignored, ~55 MB)
python dataset/dataset_creation/v2/build_index.py

# P1 — vocabulary. DO NOT run casually: see the warning below.
# python dataset/dataset_creation/v2/build_vocab.py

# P2 — one raw candidate pool per question type (writes data/candidates/, ~341 MB)
for t in existence identify_superlative relative_depth nearest_object left_right; do
    python dataset/dataset_creation/v2/$t.py
done

# P3 — balance, stratify, sanity-check, write the release CSVs
python dataset/dataset_creation/v2/build_release.py

# freeze, then regenerate the training-schema projection
python dataset/dataset_creation/v2/freeze_release.py --version v2.5
python dataset/dataset_creation/v2/export_v1_schema.py
```

**Do not re-run P1.** `data/vocab/canonical_objects.csv` is a committed,
hand-reviewed input, not a build artefact. Re-deriving it from scratch yields
148 concepts where the release ships 151, because three concepts sit within
five instances of the frequency-100 threshold. Using a regenerated vocabulary
silently changes the dataset. This is decision §13.18 in
`DATASET_CREATION_PLAN.md`.

**A P3 run reassigns every `question_id`.** They are positional, so any rebuild
invalidates an in-progress audit and requires a fresh sample. Plan the audit
after the last rebuild, not before.

**Determinism check.** Two consecutive runs from the same inputs must produce
byte-identical CSVs. If they do not, something in the pipeline is reading
corpus-wide state instead of per-scene state — the failure mode recorded in
§13.15.

---

## 6. Audit tool

Only needed if you are verifying gold labels.

```bash
conda activate kd_env

# optional: model triage, so disagreements sort first (needs DEEPSEEK_API_KEY)
python -m tools.audit_app.model_pass --types existence,left_right

# draw a fresh sample after any P3 rebuild
python -m tools.audit_app.sampling \
    --test-csv release/VQA-SUNRGBD-v2/rule_based/test.csv \
    --out audit/audit_items.csv --per-type 150 --seed 42

python -m uvicorn tools.audit_app.main:app --port 8002 --reload
# then open http://localhost:8002/  (ssh -L 8002:localhost:8002 user@server)

python -m tools.audit_app.report      # -> audit/results/report.md
```

**Start the server with `--reload`, or restart it after regenerating
`audit/audit_items.csv`.** The item list is loaded once at import into a
module global. A server started before the sample was rewritten will keep
serving the *old* sample while `/api/status` still shows the correct file path
— which is exactly how a full 750-item review once got recorded against a
stale sample. `POST /api/reload` re-reads the file without a restart.

Responses append to `audit/responses/<annotator>.jsonl`. Archive that file
before starting a review against a new release version, or two samples end up
interleaved in one log.

---

## 7. Training and evaluation

Model ids, as used in the distillation scripts:

| Role | Hugging Face id |
|---|---|
| Student (0.5B) | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| Teacher (7B) | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| Processor | `llava-hf/llava-onevision-qwen2-7b-ov-hf` (7B id, for both) |

Some datamodules reference the original `lmms-lab/llava-onevision-qwen2-7b-ov`
weights instead of the `llava-hf/*-hf` Transformers ports. They load through
different APIs — check which one a given script uses before reporting numbers
from it.

Checkpoints are written under `$MAIN_ROOT_DATA_DIR/checkpoints/`:
`baseline7b_rgb/`, `baseline_rgb/`, `kd_checkpoints/`. They are not in git; copy
them across manually if you want to resume rather than retrain. Several
training scripts resume from a hardcoded checkpoint filename, so check the
`checkpoint_filename` variable near the top of `train_online_kd.py` before
launching.

Scoring a run:

```bash
# produce predictions: greedy, temperature 0, max_new_tokens 16,
# question + "Answer with a single word or number."
# write a CSV with columns: question_id,prediction

python evaluate.py --predictions runs/my_model.csv --model-name "my model"
python evaluate.py --predictions runs/my_model.csv --constrained
```

Unanswered items count as wrong, and the majority baseline is read off train
rather than the evaluated split. Both are deliberate (§9); do not "fix" them.

---

## 8. Verification checklist

Run these on the new server before trusting any number:

```bash
python dataset/dataset_creation/v2/freeze_release.py --verify v2.4   # no drift
python -m pytest tests/ dataset/dataset_creation/v2/tests tools/audit_app/tests -q
python evaluate.py --baselines-only                                  # 30.2/33.1/34.9
wc -l release/VQA-SUNRGBD-v2/rule_based/*.csv                        # 15279/1721/12464 with headers
ls dataset/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat                   # toolbox unpacked
python -c "import h5py, torch, shapely, inflect; print('deps ok')"
```

A mismatch in the third command means the vocabulary, the release CSVs, or the
canonicaliser differ from this machine — investigate before running anything
else, because every downstream number depends on those three.

---

## 9. Reference

* `docs/DATASET_CREATION_PLAN.md` — the specification. Authoritative; read
  before changing the pipeline. §13 records decisions so they are not
  re-litigated.
* `release/VQA-SUNRGBD-v2/DATASHEET.md` — what the dataset contains and its
  known defects.
* `release/VQA-SUNRGBD-v2/stats/report.md` — sizes, balance, drop log.
* `audit/results/report.md` — gold verification results.
* `session/HANDOFF.md` — current working state (gitignored; not on the server).
