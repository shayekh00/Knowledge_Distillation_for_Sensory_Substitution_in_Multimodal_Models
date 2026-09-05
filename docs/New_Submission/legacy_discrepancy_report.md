# Legacy result provenance — discrepancy report

**Status:** WP0 / §4 deliverable of `docs/NEW_SUBMISSION.md` · CPU-only, no GPU used
**Created:** 2026-09-05
**Companion:** [`legacy_result_provenance.csv`](legacy_result_provenance.csv) (17 rows)

Addresses R1's central objection — that the headline number could not be
reconciled with the ablation tables. The conclusion is more useful than "the
records are lost": the historical scoring convention **is** fully recoverable, and
the numbers it produces are **not** comparable to the new benchmark.

---

## 1. What survives on disk

| Artifact | Present? |
|---|---|
| Prediction CSVs | ✅ 17 files in `dataset/predictions/` |
| Results summary | ✅ `dataset/predictions/summary/results_summary.csv` |
| Scoring code | ✅ `evaluation/metric.py` |
| Model checkpoints | ❌ none (`$MAIN_ROOT_DATA_DIR/checkpoints/` is empty) |
| Code commit per run | ❌ not recorded |
| Thesis / paper / reviews | ❌ not in the repository |

Checkpoint lineage is therefore **unrecoverable**. No historical row can be
re-executed; it can only be re-scored from its stored predictions.

---

## 2. The historical convention is identified and reproduces exactly

`evaluation/metric.py:simple_accuracy_metric` compares **sets of spaCy lemmas**,
not strings:

```python
pred_tokens = {token.lemma_.lower() for token in nlp(pred)}
ref_tokens  = {token.lemma_.lower() for token in nlp(ref)}
if ref_tokens == pred_tokens:  correct += 1
```

Re-scoring all 17 files with `en_core_web_md` reproduces **every** published
`Simple_Accuracy` to within 1e-9 — 17 of 17 exact. The convention is:

> micro-average over items, with match = equality of the lemma **set**.

Strict exact-match is uniformly lower (e.g. `val_pixtral` 0.2858 vs the published
0.2966 — 20 items). The gap is the convention, not noise.

### 2.1 Three defects in that convention

1. **Set equality ignores word order.** `"chair table"` scores equal to
   `"table chair"`. For any comparison whose answer is *which* of two named
   objects, order is the entire signal.
2. **Failures are silently counted wrong.** `except Exception: continue` folds a
   crash, a NaN, and a genuinely wrong answer into one bucket.
3. **It is not the dataset's canonicaliser.** It drifts freely from
   `answer_form.canonical_answer_form`, which the v2.4 protocol fixes in place.

`neural_similarity_metric` in the same file is the static word-vector similarity
R1 objected to. It is reported beside accuracy throughout the summary. Per §18 it
is removed from the new work and must not appear as evidence of correctness.

---

## 3. The blocking finding: these results are not on this benchmark

The prediction files are from the **v1** dataset. Type taxonomies are disjoint:

| | Types |
|---|---|
| Legacy files | `Color Identification`, `Count`, `Direction`, `Object Identification`, `Proximity`, `Yes/No` |
| VQA-SUNRGBD v2.4 | `existence`, `identify_superlative`, `left_right`, `nearest_object`, `relative_depth` |

Nothing overlaps. Additional confirmation:

- `Question_Id` is a bare integer (`1, 2, 3…`); v2.4 ids are `val_000000`-style.
- Row counts are 1,858 (val) and 1,841 (test); v2.4 is 1,720 and 12,463.
- `Count` appears throughout — a type **retired** in v2 (§13.16).
- Questions include *"What is the most prominent object?"* — the retired
  `largest` variant, removed in v2.4 (§13.21).

**Consequence.** No legacy number may be placed in a table beside a v2.4 number.
All 17 rows are marked `NOT comparable to v2.4`. This is the same denominator trap
the plan flags for the copied 52.6% figure in §13.1.

---

## 4. The 0% feature-KD result is a formatting artifact

`results_kd_modeltypeLdepth_val_feature_based.csv` scores 0.0000 on every one of
the six legacy types. Inspecting the file rather than the number:

- `Model_Answer` length: **min 107, median 131, max 165 characters** — never a
  short answer.
- The outputs echo the **question** and then degenerate:

```
gold 'table'   -> "how many tables are there124 framess and paper box with books
                   dispense top relative to table leg of chair seat"
gold 'chair'   -> "is there any desk142 frame top and table with drawer relative
                   to chair seater paper dispense bin potwithtoilet"
```

- 378 of 1,858 predictions begin `"what is the most prominent object124 fra…"`.

The model is reproducing prompts, not answering. This is precisely the behaviour
predicted by defect **A3** in [`implementation_audit.md`](implementation_audit.md):
the collator masks only pad tokens, so cross-entropy trains the model to
reproduce the question along with the answer. Generation was also evidently
unbounded — nothing resembles a 16-token answer.

**This is a reason to re-run the configuration, not a finding about feature-based
KD.** Per §7.1, a defect explains a score only when the configuration is
re-executed; no causal claim is made here. But the row must not be cited as
evidence that feature distillation does not work.

---

## 5. Other integrity problems

**Two names, one run.** `results_pre-trained_rgb_val_7b.csv` and
`results_val_7b.csv` are **byte-identical** (sha256 `57278aa3fb58…`), yet appear as
two rows in the summary, both at 0.3789. A results table listing them as separate
entries would double-count one run.

**Modality and label access are not recorded anywhere** — they are inferable only
from filename conventions (`_depth`, `_rgb`, `sft`, `pre-trained`, `kd`). Every
such field in the provenance CSV is tagged `(inferred from filename)`. The plan
flags exactly this for the 21.5% teacher figure, whose input modality its table
never states.

**No headline reconciliation is possible.** The thesis figures quoted in §4.1
(47.3% val / 45.5% test) do not appear in these files. The closest stored run,
`double_troublephase3`, gives:

| | micro (lemma-set) | macro over 6 types | thesis headline |
|---|---:|---:|---:|
| val | 45.21% | 48.73% | 47.3% |
| test | 42.97% | 47.04% | 45.5% |

Neither aggregation reproduces the thesis value on either split. So the stored
predictions come from *different runs* than the thesis table — a gap that cannot
be closed without the checkpoints, which are gone.

---

## 6. Disposition

Per §4.2, every row is classified.

| Class | Rows | Disposition |
|---|---:|---|
| Convention verified, benchmark incompatible | 17 | Re-scoring reproduces the published value exactly, but on v1 items. **Historical, unverified** against v2.4. Excluded from all new comparative claims |
| Re-executable | 0 | No checkpoints, no commits |
| Headline reconciled | 0 | Thesis 47.3% / 45.5% not reproducible from any stored file |

**Actions taken:** none to the historical artifacts. The thesis and submitted paper
are preserved unchanged as historical records (§4.2).

**Actions required in the manuscript:**

1. Legacy and v2.4 numbers never share a table.
2. Remove word-vector similarity as evidence of correctness.
3. Do not cite the 0% feature-KD row as a property of feature distillation.
4. State that prior results were obtained on an earlier version of the benchmark
   with a different type taxonomy and a different matching rule.
5. Report the two duplicate 7B files as one run.

**Timebox honoured.** §4.2 allows one working day and warns against letting
checkpoint archaeology delay fresh experiments. The recoverable part is now
recovered; the unrecoverable part is unrecoverable, and the new controlled runs do
not depend on it.
