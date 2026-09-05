"""Shortcut baselines and composition tables for VQA-SUNRGBD-v2 (plan §5.3).

`evaluate.py` ships three mandatory baselines — random, train-majority, and a
TF-IDF question-only classifier. The plan is explicit that the last of these is
"an informative weak baseline, not an upper bound on language shortcuts": a
linear model over bag-of-ngrams cannot exploit structure that a memorising
predictor can. These are the stronger probes.

Every baseline here is **fitted on train only** and scored on the evaluated split
through the dataset's own canonicaliser, so its numbers sit on the same scale as
the model numbers in `evaluate.py`.

Two of them are deliberately *privileged*: `anchor_prior` and `scene_prior` read
fields a deployed model never sees (the generator's `evidence`, the scene label).
They are diagnostics that bound how much of an answer is fixed by the question's
own structure, not competitors in a results table. They are labelled as such
wherever they are reported.

Usage::

    python evaluation/shortcut_baselines.py
    python evaluation/shortcut_baselines.py --split val --markdown out.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_creation", "v2")
for _path in (PROJECT_ROOT, _V2_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd  # noqa: E402

from answer_form import answers_agree  # noqa: E402
from evaluate import (  # noqa: E402
    VOCAB_DIR,
    load_canonical_vocab,
    load_release_split,
    load_synonyms,
    parse_answer_space,
)

# Which structured field each type is conditioned on for `anchor_prior`. These
# come from the generator's own `evidence`, so the key is exactly the question's
# structured content minus the image.
ANCHOR_KEYS = {
    "existence": lambda ev, row: ev.get("concept"),
    "nearest_object": lambda ev, row: ev.get("anchor_concept"),
    "identify_superlative": lambda ev, row: row.get("variant"),
    "relative_depth": lambda ev, row: (ev.get("comparative"),
                                       tuple(sorted(parse_answer_space(row.get("answer_space"))))),
    "left_right": lambda ev, row: (ev.get("a_concept"), ev.get("b_concept")),
}


def _evidence(row) -> dict:
    try:
        parsed = json.loads(row["evidence"])
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _anchor_key(question_type: str, row):
    extractor = ANCHOR_KEYS.get(question_type)
    return None if extractor is None else extractor(_evidence(row), row)


def _majority(answers) -> str | None:
    return collections.Counter(answers).most_common(1)[0][0] if len(answers) else None


def _score(rows, predict, question_type, synonym_map, canonical_vocab) -> float:
    """Fraction correct under the shared canonicaliser. A baseline that declines
    to predict scores wrong, exactly as an unanswered model item does."""
    correct = 0
    for _, row in rows.iterrows():
        prediction = predict(row)
        correct += int(prediction is not None and answers_agree(
            prediction, row["answer"], question_type, synonym_map, canonical_vocab))
    return correct / len(rows)


def _conditional_table(train_rows, key_of) -> dict:
    """key -> most frequent train answer for that key."""
    grouped = collections.defaultdict(list)
    for _, row in train_rows.iterrows():
        key = key_of(row)
        if key is not None:
            grouped[key].append(row["answer"])
    return {key: _majority(answers) for key, answers in grouped.items()}


def shortcut_baselines(train: pd.DataFrame, gold: pd.DataFrame,
                       synonym_map: dict, canonical_vocab: dict) -> dict:
    """All shortcut baselines, per question type. Fitted on `train` throughout."""
    results = collections.defaultdict(dict)
    for question_type, rows in gold.groupby("question_type", sort=True):
        train_rows = train[train["question_type"] == question_type]
        fallback = _majority(list(train_rows["answer"]))

        # 1. Best constant answer for the type (the train-majority predictor).
        results["constant"][question_type] = _score(
            rows, lambda row: fallback, question_type, synonym_map, canonical_vocab)

        # 2. Always the first-mentioned option. Only meaningful where the answer
        #    space names the candidates in question order.
        def first_option(row):
            options = parse_answer_space(row["answer_space"])
            return options[0] if options else None

        def second_option(row):
            options = parse_answer_space(row["answer_space"])
            return options[1] if len(options) > 1 else None

        results["answer_position_first"][question_type] = _score(
            rows, first_option, question_type, synonym_map, canonical_vocab)
        results["answer_position_second"][question_type] = _score(
            rows, second_option, question_type, synonym_map, canonical_vocab)

        # 3. Memorise the exact question string, backing off to the type majority.
        #    Catches template leakage a linear model would miss.
        lookup = _conditional_table(train_rows, lambda row: row["question"])
        results["question_lookup"][question_type] = _score(
            rows, lambda row: lookup.get(row["question"], fallback),
            question_type, synonym_map, canonical_vocab)

        # 4. PRIVILEGED: scene label -> majority answer.
        scene = _conditional_table(train_rows, lambda row: row["scene_type"])
        results["scene_prior"][question_type] = _score(
            rows, lambda row: scene.get(row["scene_type"], fallback),
            question_type, synonym_map, canonical_vocab)

        # 5. PRIVILEGED: the question's structured content -> majority answer.
        anchor = _conditional_table(train_rows, lambda row: _anchor_key(question_type, row))
        results["anchor_prior"][question_type] = _score(
            rows, lambda row: anchor.get(_anchor_key(question_type, row), fallback),
            question_type, synonym_map, canonical_vocab)

        # 6. Sensor -> majority answer. Detects a sensor-composition shortcut.
        sensor = _conditional_table(train_rows, lambda row: row["sensor"])
        results["sensor_prior"][question_type] = _score(
            rows, lambda row: sensor.get(row["sensor"], fallback),
            question_type, synonym_map, canonical_vocab)

    return {name: dict(scores) for name, scores in results.items()}


def composition_tables(gold: pd.DataFrame) -> dict:
    """Sensor x type and scene-category x type counts (plan §5.3)."""
    return {
        "sensor_by_type": pd.crosstab(gold["sensor"], gold["question_type"]).to_dict(),
        "scene_type_by_type": pd.crosstab(gold["scene_type"],
                                          gold["question_type"]).to_dict(),
    }


PRIVILEGED = {"scene_prior", "anchor_prior"}


def render_markdown(baselines: dict, gold: pd.DataFrame, split: str) -> str:
    types = sorted(gold["question_type"].unique())
    lines = [f"# Shortcut baselines — `{split}` split", "",
             "Fitted on **train**, scored through the shared canonicaliser.", "",
             "| Baseline | " + " | ".join(types) + " | Macro |",
             "|---|" + "---:|" * (len(types) + 1)]
    for name, scores in baselines.items():
        marker = " ⚠️" if name in PRIVILEGED else ""
        cells = " | ".join(f"{scores[t]:.1%}" for t in types)
        macro = sum(scores[t] for t in types) / len(types)
        lines.append(f"| `{name}`{marker} | {cells} | **{macro:.1%}** |")
    lines += ["", "⚠️ = privileged diagnostic: reads fields a deployed model never sees "
              "(generator evidence, scene label). Bounds how much of the answer is fixed "
              "by question structure alone. Not a competitor in a results table.", ""]

    counts = pd.crosstab(gold["sensor"], gold["question_type"])
    lines += [f"## Sensor x question type (`{split}`)", "",
              "| Sensor | " + " | ".join(types) + " | Total |",
              "|---|" + "---:|" * (len(types) + 1)]
    for sensor, row in counts.iterrows():
        lines.append(f"| {sensor} | " + " | ".join(str(int(row[t])) for t in types)
                     + f" | {int(row.sum())} |")
    lines += ["", "Row shares differing markedly across types would mean a type is "
              "disproportionately drawn from one sensor's depth characteristics.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--markdown", help="Write the report here as well as to stdout.")
    parser.add_argument("--json", dest="json_path", help="Write machine-readable results here.")
    args = parser.parse_args()

    synonym_map = load_synonyms(os.path.join(VOCAB_DIR, "synonyms.csv"))
    canonical_vocab = load_canonical_vocab(os.path.join(VOCAB_DIR, "canonical_objects.csv"))
    gold = load_release_split(args.split)
    train = load_release_split("train")

    baselines = shortcut_baselines(train, gold, synonym_map, canonical_vocab)
    report = render_markdown(baselines, gold, args.split)
    print(report)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({"split": args.split, "baselines": baselines,
                       "composition": composition_tables(gold)}, handle, indent=2)


if __name__ == "__main__":
    main()
