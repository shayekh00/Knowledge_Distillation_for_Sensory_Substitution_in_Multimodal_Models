# Shortcut baselines and split composition

**Status:** §5.3 deliverable of `docs/NEW_SUBMISSION.md` · CPU-only, no GPU used
**Created:** 2026-09-05
**Regenerate with:** `python evaluation/shortcut_baselines.py --split test`
**Companion:** [`dataset_protocol.md`](dataset_protocol.md), [`experiment_protocol.md`](experiment_protocol.md)

`evaluate.py` ships three baselines — random, train-majority, and a TF-IDF
question-only classifier. The plan is explicit that the last is *"an informative
weak baseline, not an upper bound on language shortcuts."* These are the stronger
probes it asks for: lexical, answer-position, anchor-conditioned, and scene-prior,
all **fitted on train only** and scored through the shared canonicaliser.

---

## 1. Results (test split)

| Baseline | existence | identify_superlative | left_right | nearest_object | relative_depth | Macro |
|---|---:|---:|---:|---:|---:|---:|
| `constant` (train majority) | 50.0% | 8.0% | 50.0% | 7.8% | 9.2% | **25.0%** |
| `answer_position_first` | 50.0% | 0.0% | 50.0% | 0.0% | 49.8% | **30.0%** |
| `answer_position_second` | 50.0% | 0.0% | 50.0% | 0.0% | 50.2% | **30.0%** |
| `question_lookup` | 50.6% | 10.2% | 50.5% | 8.9% | 17.8% | **27.6%** |
| `scene_prior` ⚠️ | **60.5%** | 14.4% | 49.9% | 9.2% | 12.5% | **29.3%** |
| `anchor_prior` ⚠️ | 50.0% | 12.0% | 50.3% | 11.0% | 40.7% | **32.8%** |
| `sensor_prior` | 53.4% | 8.0% | 50.7% | 8.3% | 9.2% | **25.9%** |

⚠️ **Privileged diagnostics.** `scene_prior` and `anchor_prior` read fields a
deployed model never receives (the scene label; the generator's `evidence`). They
bound how much of an answer is fixed by question structure alone. They are not
competitors in a results table — with one important qualification in §3.

For reference, the repaired `evaluate.py` baselines on the same split: chance
30.3%, random 30.4%, train-majority 33.1%, TF-IDF question-only 34.9%.

---

## 2. What comes out clean

**No answer-position shortcut.** `relative_depth` is the only type whose answer
space names candidates in question order, and always-first scores **49.8%**
against always-second **50.2%** — a 0.4-point spread on 2,471 items. The
generator's per-pair coin flip on mention order works. A model cannot profit from
position.

**The object pair does not determine the answer.** `anchor_prior` conditions
`relative_depth` on `(comparative, sorted object pair)` learned from train and
scores **40.7%** — *below* the 50% chance floor. Knowing that the question
compares a chair and a table, and in which direction, is actively misleading
without the image. This is the strongest single piece of evidence that the type
requires visual depth.

**The anchor cap holds.** `nearest_object` conditioned on its anchor concept gives
**11.0%** against 7.8% constant — a 3.2-point edge, consistent with the 20%
anchor-conditional cap that `balance.py` enforces at build time.

**Question memorisation is weak.** Exact question-string lookup buys 0.6 points on
`existence` and 0.5 on `left_right`. Its one real gain is `relative_depth`
(17.8% vs 9.2%), which is expected and harmless: the question names both objects,
so the lookup is partly recovering the answer *space*, not the answer — and it
still lands far below the 50% chance floor for that type.

**`left_right` is clean across every probe** — 49.9% to 50.7% throughout. No
lexical, scene, sensor, or positional signal.

---

## 3. The finding that needs disclosing: scene type leaks `existence`

`existence` is balanced to exactly 50/50 **globally** (train 50.0%, test 50.0%)
and **per concept** — every concept has `yes == no`, verified directly. It is
**not** balanced within scene type:

| scene_type | train n | train yes | test n | test yes | prior predicts | accuracy |
|---|---:|---:|---:|---:|:--:|---:|
| bedroom | 465 | 40.6% | 271 | 26.9% | `no` | **73.1%** |
| classroom | 436 | 55.3% | 262 | 52.3% | `yes` | 52.3% |
| office | 451 | 56.8% | 254 | 54.3% | `yes` | 54.3% |
| rest_space | 398 | 59.3% | 222 | 59.0% | `yes` | 59.0% |
| furniture_store | 507 | 38.1% | 186 | 45.7% | `no` | 54.3% |

Across all scene types the label alone reaches **60.5%** against a 50.0%
balanced baseline — **+10.5 points from the room category and nothing else.**

**Why this is not merely a privileged diagnostic.** Scene type is not an
annotation a model is denied — it is *visually obvious*. A VLM that recognises a
bedroom and answers `no` to everything scores 73.1% on bedroom existence items
without performing any object reasoning at all. Unlike `anchor_prior`, this
shortcut is available to a real model through the image.

**It compounds with a known limitation.** `dataset_protocol.md` §7.1 records that
existence negatives are decided on canonical name alone, and §7.2 that `table` is
74% of test existence items. A bedroom rarely contains a canonical `table`, so
"bedroom ⇒ no" is *usually right for the wrong reason* — and is right precisely
where the gold label is most likely to have been decided against an annotated
`desk` or `nightstand`.

**A distribution shift sits inside it.** Bedroom yes-rate is 40.6% in train but
26.9% in test — a 13.7-point drift for the same scene category, so this is not a
stable prior a model could simply learn and rely on.

### Required actions

1. **Report `scene_prior` beside the mandatory baselines** for `existence` in the
   paper. A reader comparing a model at, say, 62% against a 50% majority baseline
   is being shown the wrong reference point.
2. **Report `existence` per scene type**, not only in aggregate.
3. Do not rebuild or rebalance to remove this — the release is frozen
   (`experiment_protocol.md` §3.2). Disclose it.
4. Treat any `existence` gain under ~60% as **unevidenced** for visual reasoning.

`sensor_prior` shows a smaller version of the same effect (53.4% on existence,
+3.4 points). Worth a sentence, not a section.

---

## 4. Split composition

### Sensor × question type (test)

| Sensor | existence | identify_superlative | left_right | nearest_object | relative_depth | Total |
|---|---:|---:|---:|---:|---:|---:|
| kv1 | 480 | 428 | 518 | 537 | 541 | 2,504 |
| kv2 | 945 | 941 | 845 | 855 | 865 | 4,451 |
| realsense | 284 | 282 | 225 | 245 | 220 | 1,256 |
| xtion | 849 | 839 | 882 | 837 | 845 | 4,252 |

Composition is even within each sensor — no type is disproportionately drawn from
one sensor's depth characteristics, which is what `stratified_subsample` is for.
This matters directly for the **xtion depth-decoding divergence** in
`implementation_audit.md` §A2: xtion is 34% of the test split, so a decoder that
is wrong on xtion is wrong on roughly a third of every reported number.

The full scene-category × type table is in the `--json` output; it is long-tailed
and belongs in the supplement rather than the main paper.

---

## 5. Reproducibility

```bash
python evaluation/shortcut_baselines.py --split test \
    --markdown docs/New_Submission/shortcut_baselines_test.md \
    --json runs/shortcut_baselines_test.json
```

Deterministic — every baseline is a counting argument over train, with no sampling
and no seed. Covered by `tests/test_shortcut_baselines.py`.
