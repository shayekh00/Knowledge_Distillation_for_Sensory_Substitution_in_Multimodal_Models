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
    accuracy: float
    macro_f1: float | None


def load_release_split(split: str, release_dir: str = RELEASE_DIR) -> pd.DataFrame:
    path = os.path.join(release_dir, f"{split}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} does not exist — is the release frozen?")
    return pd.read_csv(path)


def load_predictions(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"question_id", "prediction"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    return frame.drop_duplicates(subset="question_id", keep="last")


def snap_to_answer_space(prediction: str, answer_space: str, question_type: str,
                         synonym_map: dict, canonical_vocab: dict) -> str:
    """Constrained decoding, approximated after the fact.

    True constrained decoding restricts generation to the answer space, so the
    model cannot produce an illegal string at all. Scoring a free-form run
    afterwards can only approximate that, in three steps: take an exact
    canonical match; else the option mentioned earliest in the answer
    ("the right side" -> `right`), since a model that names an option is
    choosing it; else the first option, so a constrained model always commits
    rather than abstaining.

    Earliest-mention rather than first-listed matters for `relative_depth`,
    where both options appear in the question and a verbose answer may repeat
    both ("the chair is closer than the table").

    Open-vocabulary types declare no `answer_space` and are returned untouched.
    Constraining them would mean snapping onto the 151-concept vocabulary, and
    the fallback — commit to *some* legal concept — has no meaning when the
    options are not the two the question named; it would just award or destroy
    accuracy at random. Constrained decoding is therefore reported for the
    closed types only, which is where R1 asked for it.
    """
    if answer_space is None or (isinstance(answer_space, float) and answer_space != answer_space):
        return prediction  # NaN: no declared answer space
    options = [option for option in str(answer_space).split("|") if option and option != "nan"]
    if not options:
        return prediction
    for option in options:
        if answers_agree(prediction, option, question_type, synonym_map, canonical_vocab):
            return option
    normalized = " " + str(prediction or "").strip().lower().replace("_", " ") + " "
    mentioned = [(normalized.find(f" {option.lower()} "), option) for option in options]
    mentioned = [(position, option) for position, option in mentioned if position >= 0]
    return min(mentioned)[1] if mentioned else options[0]


def score_predictions(gold: pd.DataFrame, predictions: pd.DataFrame,
                      synonym_map: dict, canonical_vocab: dict,
                      constrained: bool = False) -> list[TypeScore]:
    """Per-type exact-match accuracy, and macro-F1 for the closed types.

    An item with no prediction counts as wrong rather than being dropped:
    silently scoring only the rows a model answered would reward abstention.
    """
    predicted_by_id = dict(zip(predictions["question_id"], predictions["prediction"]))
    scores = []
    for question_type, rows in gold.groupby("question_type", sort=True):
        correct, gold_labels, predicted_labels, n_predicted = 0, [], [], 0
        for _, row in rows.iterrows():
            prediction = predicted_by_id.get(row["question_id"])
            has_prediction = prediction is not None and str(prediction).strip() != ""
            n_predicted += int(has_prediction)
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
            accuracy=correct / len(rows),
            macro_f1=(float(f1_score(gold_labels, predicted_labels, average="macro",
                                     zero_division=0)) if question_type in CLOSED_TYPES else None),
        ))
    return scores


def _closed_type_prediction_label(question_type: str, row, prediction, has_prediction: bool,
                                  synonym_map: dict, canonical_vocab: dict) -> str:
    if not has_prediction:
        return "<none>"
    if question_type != "relative_depth":
        return canonical_answer_form(prediction, question_type, synonym_map, canonical_vocab)
    options = str(row["answer_space"]).split("|")
    for position, option in zip(("first", "second"), options):
        if answers_agree(prediction, option, question_type, synonym_map, canonical_vocab):
            return position
    return "<none>"


def macro_accuracy(scores: list[TypeScore]) -> float:
    """Headline number: unweighted mean over types, so a large type cannot
    dominate the score of a benchmark whose types differ in size."""
    return float(np.mean([score.accuracy for score in scores])) if scores else 0.0


def random_baseline(gold: pd.DataFrame, canonical_vocab: dict, seed: int = DEFAULT_SEED) -> dict:
    """Uniform over the row's declared answer space, or over the answerable
    vocabulary for open types."""
    rng = np.random.default_rng(seed)
    open_vocabulary = sorted(concept for concept, entry in canonical_vocab.items()
                             if not entry["is_structural"])
    accuracies = {}
    for question_type, rows in gold.groupby("question_type", sort=True):
        correct = 0
        for _, row in rows.iterrows():
            options = [option for option in str(row["answer_space"] or "").split("|") if option]
            if not options:
                options = open_vocabulary
            correct += int(options[int(rng.integers(len(options)))] == row["answer"])
        accuracies[question_type] = correct / len(rows)
    return accuracies


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
    lines += ["| Type | n | Random | Majority | Question-only |"
              + (" Model | Macro-F1 |" if scores else ""),
              "|---|---:|---:|---:|---:|" + ("---:|---:|" if scores else "")]
    score_by_type = {score.question_type: score for score in (scores or [])}
    for question_type in types:
        score = score_by_type.get(question_type)
        row = (f"| {question_type} | {score.n_items if score else '—'} "
               f"| {baselines['random'][question_type]:.1%} "
               f"| {baselines['majority'][question_type]:.1%} "
               f"| {baselines['question_only'][question_type]:.1%} |")
        if scores:
            f1 = f"{score.macro_f1:.3f}" if score.macro_f1 is not None else "—"
            row += f" {score.accuracy:.1%} | {f1} |"
        lines.append(row)
    lines += ["", f"Macro accuracy (random): {np.mean(list(baselines['random'].values())):.1%}",
              f"Macro accuracy (majority): {np.mean(list(baselines['majority'].values())):.1%}",
              f"Macro accuracy (question-only): {np.mean(list(baselines['question_only'].values())):.1%}"]
    if scores:
        lines += [f"**Macro accuracy (model): {macro_accuracy(scores):.1%}**", "",
                  f"Items answered: {sum(s.n_predicted for s in scores)}"
                  f" / {sum(s.n_items for s in scores)} (unanswered items count as wrong)."]
    lines += ["", "Macro-F1 is reported for closed answer spaces only; for `relative_depth` the "
              "classes are answer *position* (first / second named object), because its answer "
              "space is item-specific.", ""]
    return "\n".join(lines)


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
        "random": random_baseline(gold, canonical_vocab, args.seed),
        "majority": majority_baseline(train, gold),
        "question_only": question_only_baseline(train, gold, args.seed),
    }
    scores = None
    if args.predictions:
        scores = score_predictions(gold, load_predictions(args.predictions),
                                   synonym_map, canonical_vocab, args.constrained)

    report = render_markdown(scores, baselines, args.split, args.constrained, args.model_name)
    print(report)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report)
    if args.json_path:
        payload = {"split": args.split, "seed": args.seed, "constrained": args.constrained,
                   "baselines": baselines,
                   "per_type": [asdict(score) for score in scores] if scores else None,
                   "macro_accuracy": macro_accuracy(scores) if scores else None}
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
