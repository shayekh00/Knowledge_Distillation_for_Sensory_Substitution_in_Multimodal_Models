"""Evaluation protocol shipped with VQA-SUNRGBD-v2 (DATASET_CREATION_PLAN.md §9).

Papers using this benchmark are asked to score with this file, so that numbers
from different models are comparable. What it fixes in place:

* one canonicaliser for gold and predictions (`answer_form`), so a model is
  never scored wrong for a surface form the dataset itself would accept;
* exact-match accuracy per type, macro accuracy over types as the headline,
  and macro-F1 over answer classes for the closed types;
* the three mandatory baselines — random, majority class, and question-only —
  reported in the same table, so a reader can see how much of any score is
  language prior rather than perception;
* optional constrained decoding, which snaps a prediction onto the row's own
  answer space and is reported as an extra column, never as the headline.

Decoding itself is the caller's job: run the model greedily (temperature 0,
max_new_tokens 16, the Rule Q5 prompt suffix) and write a CSV of
`question_id,prediction`. This file does not load models, so it stays runnable
on CPU in seconds and its numbers are reproducible from the repository alone.

Usage::

    python evaluate.py --predictions runs/onevision_7b.csv
    python evaluate.py --predictions runs/onevision_7b.csv --constrained
    python evaluate.py --baselines-only --markdown release/.../stats/baselines.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_creation", "v2")
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from answer_form import answers_agree, canonical_answer_form  # noqa: E402
from question_only import evaluate_question_type, target_label  # noqa: E402
from vocab import load_canonical_vocab, load_synonyms  # noqa: E402

RELEASE_DIR = os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2", "rule_based")
VOCAB_DIR = os.path.join(PROJECT_ROOT, "data", "vocab")
DEFAULT_SEED = 42

# Types with a small fixed answer space, where macro-F1 over answer classes is
# meaningful. Open-vocabulary types have ~130 answer classes with long tails;
# a macro-F1 there is dominated by classes with a handful of items.
CLOSED_TYPES = ("existence", "left_right", "relative_depth")


@dataclass(frozen=True)
class TypeScore:
    question_type: str
    n_items: int
    n_predicted: int
    n_invalid: int
    accuracy: float
    macro_f1: float | None


def is_missing(value) -> bool:
    """True for an absent answer: None, a pandas/NumPy NaN, or blank text.

    The NaN case has to be tested *before* `str()`: `str(float("nan"))` is
    `"nan"`, a perfectly non-empty string, so a NaN prediction would otherwise
    be counted as an item the model successfully answered.
    """
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() == ""


def parse_answer_space(answer_space) -> list[str]:
    """The options a row declares, or `[]` when it declares none.

    Open-vocabulary rows carry NaN in this column. `NaN` is *truthy* in Python,
    so the obvious `answer_space or ""` guard never fires: `str(NaN)` yields the
    literal string `"nan"`, and every caller downstream then treats `"nan"` as
    the row's one and only legal answer. That is exactly the defect that made
    the random baseline score 0.0% on both open-vocabulary types.
    """
    if is_missing(answer_space):
        return []
    options = [option.strip() for option in str(answer_space).split("|")]
    return [option for option in options if option and option.lower() != "nan"]


def answerable_vocabulary(canonical_vocab: dict) -> list[str]:
    """Display forms of every non-structural concept — the eligible answer set
    for the open-vocabulary types, sorted so sampling from it is reproducible."""
    return sorted(entry["display_name"].replace("_", " ")
                  for entry in canonical_vocab.values() if not entry["is_structural"])


def is_legal_answer(prediction, row, question_type: str, synonym_map: dict,
                    canonical_vocab: dict, legal_open_answers: set | None = None) -> bool:
    """Whether a prediction is a *legal* answer for this row — not whether it is
    correct. Closed rows must name a declared option; open rows must name a
    non-structural concept in the vocabulary.

    Membership is tested on the canonical *display* form, not on
    `canonicalize()["in_vocab"]`. The vocabulary is keyed by concept
    (`tissuebox`) while the release answers carry the display form
    (`tissue box`), so `in_vocab` is False for every multiword concept — it
    would flag 146 `trash can` and 48 `file cabinet` gold answers in the test
    split as invalid.
    """
    options = parse_answer_space(row["answer_space"])
    if options:
        return any(answers_agree(prediction, option, question_type, synonym_map, canonical_vocab)
                   for option in options)
    if legal_open_answers is None:
        legal_open_answers = set(answerable_vocabulary(canonical_vocab))
    return canonical_answer_form(prediction, question_type,
                                 synonym_map, canonical_vocab) in legal_open_answers


def load_release_split(split: str, release_dir: str = RELEASE_DIR) -> pd.DataFrame:
    path = os.path.join(release_dir, f"{split}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} does not exist — is the release frozen?")
    return pd.read_csv(path)


def load_predictions(path: str) -> pd.DataFrame:
    """Load a prediction file, refusing anything ambiguous.

    A duplicated `question_id` used to be resolved by silently keeping the last
    row. That hides a real defect in a generation run — two different answers
    for one item — and makes the reported denominator unverifiable, so it is now
    an error the caller has to fix.
    """
    frame = pd.read_csv(path)
    missing = {"question_id", "prediction"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    duplicated = frame.loc[frame["question_id"].duplicated(), "question_id"].unique()
    if len(duplicated):
        shown = ", ".join(str(question_id) for question_id in duplicated[:5])
        raise SystemExit(
            f"{path} contains {len(duplicated)} duplicated question_id(s): {shown}"
            + (", ..." if len(duplicated) > 5 else "")
            + ". Refusing to guess which row was intended — deduplicate the file.")
    return frame


def prediction_id_report(gold: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Which gold items went unanswered, and which predicted ids are not in the
    evaluated split. Unexpected ids are reported, never scored."""
    gold_ids = set(gold["question_id"])
    predicted_ids = set(predictions["question_id"])
    return {
        "n_gold_items": len(gold_ids),
        "n_prediction_rows": int(len(predictions)),
        "n_missing": len(gold_ids - predicted_ids),
        "n_unexpected": len(predicted_ids - gold_ids),
        "unexpected_examples": sorted(predicted_ids - gold_ids)[:5],
    }


_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")


def _mention_form(text) -> str:
    """Whitespace-delimited form used to look for an option inside a sentence.

    Punctuation has to collapse to spaces, not be left in place: searching for
    `" left "` inside `"it is on the left."` fails on the trailing period, which
    is precisely the case a verbose model produces. That miss used to be hidden
    by the `options[0]` fallback, which returned the right answer for the wrong
    reason whenever the intended option happened to be listed first.
    """
    return _NON_ALPHANUMERIC_RE.sub(" ", str(text or "").lower()).strip()


def snap_to_answer_space(prediction: str, answer_space: str, question_type: str,
                         synonym_map: dict, canonical_vocab: dict) -> str:
    """Constrained decoding, approximated after the fact.

    True constrained decoding restricts generation to the answer space, so the
    model cannot produce an illegal string at all. Scoring a free-form run
    afterwards can only approximate that, in two steps: take an exact canonical
    match; else the option mentioned earliest in the answer ("the right side" ->
    `right`), since a model that names an option is choosing it.

    Earliest-mention rather than first-listed matters for `relative_depth`,
    where both options appear in the question and a verbose answer may repeat
    both ("the chair is closer than the table").

    A response that names no option at all is returned **unchanged**, and so
    scores wrong. It used to be snapped onto `options[0]`, which manufactured a
    decision the model never made: an unparsable answer is not evidence that it
    selected the first option, and on a binary type that fallback hands out
    accuracy at chance for free (plan §6.3).

    Open-vocabulary types declare no `answer_space` and are returned untouched.
    Constraining them would mean snapping onto the 151-concept vocabulary, and
    committing to *some* legal concept has no meaning when the options are not
    the two the question named. Constrained decoding is therefore reported for
    the closed types only, which is where R1 asked for it.
    """
    options = parse_answer_space(answer_space)
    if not options:
        return prediction
    for option in options:
        if answers_agree(prediction, option, question_type, synonym_map, canonical_vocab):
            return option
    normalized = f" {_mention_form(prediction)} "
    mentioned = [(normalized.find(f" {_mention_form(option)} "), option) for option in options]
    mentioned = [(position, option) for position, option in mentioned if position >= 0]
    return min(mentioned)[1] if mentioned else prediction


def score_predictions(gold: pd.DataFrame, predictions: pd.DataFrame,
                      synonym_map: dict, canonical_vocab: dict,
                      constrained: bool = False) -> list[TypeScore]:
    """Per-type exact-match accuracy, and macro-F1 for the closed types.

    An item with no prediction counts as wrong rather than being dropped:
    silently scoring only the rows a model answered would reward abstention.
    """
    predicted_by_id = dict(zip(predictions["question_id"], predictions["prediction"]))
    legal_open_answers = set(answerable_vocabulary(canonical_vocab))
    scores = []
    for question_type, rows in gold.groupby("question_type", sort=True):
        correct, gold_labels, predicted_labels, n_predicted, n_invalid = 0, [], [], 0, 0
        for _, row in rows.iterrows():
            prediction = predicted_by_id.get(row["question_id"])
            has_prediction = not is_missing(prediction)
            n_predicted += int(has_prediction)
            if has_prediction and not is_legal_answer(prediction, row, question_type,
                                                      synonym_map, canonical_vocab,
                                                      legal_open_answers):
                n_invalid += 1
            if has_prediction and constrained:
                prediction = snap_to_answer_space(
                    prediction, row["answer_space"], question_type, synonym_map, canonical_vocab)
            is_correct = has_prediction and answers_agree(
                prediction, row["answer"], question_type, synonym_map, canonical_vocab)
            correct += int(is_correct)
            if question_type in CLOSED_TYPES:
                gold_labels.append(target_label(question_type, row))
                # A prediction that is wrong or absent still needs a label for
                # F1; mapping it to the gold label's complement would fake a
                # confusion matrix, so use the prediction's own canonical form.
                predicted_labels.append(
                    _closed_type_prediction_label(question_type, row, prediction, has_prediction,
                                                  synonym_map, canonical_vocab))
        scores.append(TypeScore(
            question_type=question_type,
            n_items=len(rows),
            n_predicted=n_predicted,
            n_invalid=n_invalid,
            accuracy=correct / len(rows),
            # `labels=` pins the averaged classes to the ones the benchmark
            # defines, so the <invalid> sentinel is never itself scored.
            macro_f1=(float(f1_score(gold_labels, predicted_labels,
                                     labels=sorted(set(gold_labels)), average="macro",
                                     zero_division=0)) if question_type in CLOSED_TYPES else None),
        ))
    return scores


INVALID_LABEL = "<invalid>"


def _closed_type_prediction_label(question_type: str, row, prediction, has_prediction: bool,
                                  synonym_map: dict, canonical_vocab: dict) -> str:
    """The F1 label for a prediction on a closed type.

    Anything that is not one of the row's declared options collapses to a single
    `<invalid>` sentinel rather than to its own canonical form. Letting arbitrary
    generated strings through would mint a fresh F1 class per hallucination
    ("banana"), and macro-F1 would then average over classes the benchmark never
    defined. The sentinel is excluded from the averaged label set by
    `score_predictions`, so an invalid answer costs the true class a false
    negative and nothing else (plan §6.1).
    """
    if not has_prediction:
        return INVALID_LABEL
    options = parse_answer_space(row["answer_space"])
    if question_type == "relative_depth":
        for position, option in zip(("first", "second"), options):
            if answers_agree(prediction, option, question_type, synonym_map, canonical_vocab):
                return position
        return INVALID_LABEL
    for option in options:
        if answers_agree(prediction, option, question_type, synonym_map, canonical_vocab):
            return canonical_answer_form(option, question_type, synonym_map, canonical_vocab)
    return INVALID_LABEL


def macro_accuracy(scores: list[TypeScore]) -> float:
    """Headline number: unweighted mean over types, so a large type cannot
    dominate the score of a benchmark whose types differ in size."""
    return float(np.mean([score.accuracy for score in scores])) if scores else 0.0


def random_baseline(gold: pd.DataFrame, synonym_map: dict, canonical_vocab: dict,
                    seed: int = DEFAULT_SEED) -> dict:
    """Seeded empirical random guessing: uniform over the row's declared answer
    space, or over the answerable vocabulary for the open types.

    Two defects used to make this score 0.0% on both open-vocabulary types:
    `str(answer_space or "")` yielded `"nan"` for their NaN answer space (NaN is
    truthy), so every draw guessed the literal string "nan"; and the draw was
    compared to gold with `==`, bypassing the shared canonicaliser that every
    other comparison in this file goes through.
    """
    rng = np.random.default_rng(seed)
    open_vocabulary = answerable_vocabulary(canonical_vocab)
    accuracies = {}
    for question_type, rows in gold.groupby("question_type", sort=True):
        correct = 0
        for _, row in rows.iterrows():
            options = parse_answer_space(row["answer_space"]) or open_vocabulary
            sampled = options[int(rng.integers(len(options)))]
            correct += int(answers_agree(sampled, row["answer"], question_type,
                                         synonym_map, canonical_vocab))
        accuracies[question_type] = correct / len(rows)
    return accuracies


def theoretical_chance(gold: pd.DataFrame, canonical_vocab: dict) -> dict:
    """Uniform chance computed rather than sampled: the mean of 1/|options| over
    the rows of each type.

    Reported beside the seeded empirical baseline because the two answer
    different questions, and because the per-type value is the honest floor:
    `relative_depth` chance is 50% over the two objects that row names, not
    ~1/151 over every object name in the vocabulary (plan §6.2).
    """
    open_size = len(answerable_vocabulary(canonical_vocab))
    chances = {}
    for question_type, rows in gold.groupby("question_type", sort=True):
        total = 0.0
        for _, row in rows.iterrows():
            options = parse_answer_space(row["answer_space"])
            total += 1.0 / (len(options) if options else open_size)
        chances[question_type] = total / len(rows)
    return chances


def split_majority_share(gold: pd.DataFrame) -> dict:
    """Share of the *evaluated* split held by its own most frequent answer.

    This is a descriptive oracle statistic, not a baseline a model could reach
    without seeing the split's labels. It is reported separately from
    `majority_baseline`, which fits on train, so the two are never confused
    (plan §6.2).
    """
    shares = {}
    for question_type, rows in gold.groupby("question_type", sort=True):
        labels = [target_label(question_type, row) for _, row in rows.iterrows()]
        shares[question_type] = float(pd.Series(labels).value_counts().max() / len(labels))
    return shares


def majority_baseline(train: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Most frequent *train* answer per type, scored on the evaluated split.

    Taken from train rather than from the evaluated split itself: a majority
    read off the test set is an oracle, and would understate how much a real
    language prior buys.
    """
    accuracies = {}
    for question_type, rows in gold.groupby("question_type", sort=True):
        train_rows = train[train["question_type"] == question_type]
        if train_rows.empty:
            accuracies[question_type] = 0.0
            continue
        gold_labels = [target_label(question_type, row) for _, row in rows.iterrows()]
        train_labels = [target_label(question_type, row) for _, row in train_rows.iterrows()]
        majority = pd.Series(train_labels).value_counts().idxmax()
        accuracies[question_type] = float(np.mean([label == majority for label in gold_labels]))
    return accuracies


def question_only_baseline(train: pd.DataFrame, gold: pd.DataFrame,
                           seed: int = DEFAULT_SEED) -> dict:
    """TF-IDF + logistic regression on the question text alone (§8.4)."""
    accuracies = {}
    for question_type, rows in gold.groupby("question_type", sort=True):
        train_rows = train[train["question_type"] == question_type]
        result = evaluate_question_type(question_type, train_rows, rows, random_state=seed)
        accuracies[question_type] = result["accuracy"]
    return accuracies


def render_markdown(scores: list[TypeScore] | None, baselines: dict, split: str,
                    constrained: bool, model_name: str) -> str:
    types = ([score.question_type for score in scores] if scores
             else sorted(baselines["random"]))
    lines = [f"# VQA-SUNRGBD-v2 — evaluation on `{split}`", ""]
    if scores:
        lines += [f"Model: **{model_name}**"
                  + ("  ·  constrained decoding over the answer space" if constrained else ""), ""]
    lines += ["| Type | n | Chance | Random | Majority | Question-only |"
              + (" Model | Macro-F1 | Invalid |" if scores else ""),
              "|---|---:|---:|---:|---:|---:|" + ("---:|---:|---:|" if scores else "")]
    score_by_type = {score.question_type: score for score in (scores or [])}
    for question_type in types:
        score = score_by_type.get(question_type)
        row = (f"| {question_type} | {score.n_items if score else '—'} "
               f"| {baselines['chance'][question_type]:.1%} "
               f"| {baselines['random'][question_type]:.1%} "
               f"| {baselines['majority'][question_type]:.1%} "
               f"| {baselines['question_only'][question_type]:.1%} |")
        if scores:
            f1 = f"{score.macro_f1:.3f}" if score.macro_f1 is not None else "—"
            invalid = score.n_invalid / score.n_items if score.n_items else 0.0
            row += f" {score.accuracy:.1%} | {f1} | {invalid:.1%} |"
        lines.append(row)
    lines += ["", f"Macro accuracy (chance): {np.mean(list(baselines['chance'].values())):.1%}",
              f"Macro accuracy (random): {np.mean(list(baselines['random'].values())):.1%}",
              f"Macro accuracy (majority): {np.mean(list(baselines['majority'].values())):.1%}",
              f"Macro accuracy (question-only): {np.mean(list(baselines['question_only'].values())):.1%}"]
    if scores:
        lines += [f"**Macro accuracy (model): {macro_accuracy(scores):.1%}**", "",
                  f"Items answered: {sum(s.n_predicted for s in scores)}"
                  f" / {sum(s.n_items for s in scores)} (unanswered items count as wrong)."
                  f"  Invalid answers: {sum(s.n_invalid for s in scores)}"
                  f" / {sum(s.n_items for s in scores)}."]
    lines += ["", "Macro-F1 is reported for closed answer spaces only; for `relative_depth` the "
              "classes are answer *position* (first / second named object), because its answer "
              "space is item-specific.",
              "",
              "`Chance` is computed uniform chance (mean of 1/|options|); `Random` is a seeded "
              "empirical draw. `Majority` is fitted on **train**; the evaluated split's own "
              "majority share is an oracle statistic and is reported in the JSON output only.",
              ""]
    return "\n".join(lines)


def check_reported_aggregates(scores: list[TypeScore], reported_macro: float) -> None:
    """The printed macro must equal the mean of the printed per-type numbers.

    R1 could not reconcile the previous submission's headline with its own
    ablation tables. This makes that class of mismatch impossible to print.
    """
    if not scores:
        return
    recomputed = float(np.mean([score.accuracy for score in scores]))
    if abs(recomputed - reported_macro) > 1e-9:
        raise SystemExit(
            f"Aggregate check failed: reported macro {reported_macro:.6f} does not equal the "
            f"mean of the per-type accuracies {recomputed:.6f}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", help="CSV with columns question_id,prediction.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--constrained", action="store_true",
                        help="Snap each prediction onto its row's answer space.")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--model-name", default="unnamed model")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--markdown", help="Write the report here as well as to stdout.")
    parser.add_argument("--json", dest="json_path", help="Write machine-readable results here.")
    args = parser.parse_args()

    if not args.predictions and not args.baselines_only:
        raise SystemExit("Pass --predictions, or --baselines-only for the baseline table.")

    synonym_map = load_synonyms(os.path.join(VOCAB_DIR, "synonyms.csv"))
    canonical_vocab = load_canonical_vocab(os.path.join(VOCAB_DIR, "canonical_objects.csv"))
    gold = load_release_split(args.split)
    train = load_release_split("train")

    baselines = {
        "chance": theoretical_chance(gold, canonical_vocab),
        "random": random_baseline(gold, synonym_map, canonical_vocab, args.seed),
        "majority": majority_baseline(train, gold),
        "question_only": question_only_baseline(train, gold, args.seed),
    }
    scores, id_report = None, None
    if args.predictions:
        predictions = load_predictions(args.predictions)
        id_report = prediction_id_report(gold, predictions)
        if id_report["n_unexpected"]:
            examples = ", ".join(str(item) for item in id_report["unexpected_examples"])
            print(f"warning: {id_report['n_unexpected']} predicted question_id(s) are not in "
                  f"the '{args.split}' split and were ignored: {examples}", file=sys.stderr)
        scores = score_predictions(gold, predictions, synonym_map, canonical_vocab,
                                   args.constrained)
        check_reported_aggregates(scores, macro_accuracy(scores))

    report = render_markdown(scores, baselines, args.split, args.constrained, args.model_name)
    print(report)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report)
    if args.json_path:
        payload = {"split": args.split, "seed": args.seed, "constrained": args.constrained,
                   "baselines": baselines,
                   "split_majority_share_oracle": split_majority_share(gold),
                   "prediction_ids": id_report,
                   "per_type": [asdict(score) for score in scores] if scores else None,
                   "macro_accuracy": macro_accuracy(scores) if scores else None}
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
