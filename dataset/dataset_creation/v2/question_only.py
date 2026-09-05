"""Question-only baseline used as a release-blocking language-leakage test."""
from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MAX_EXCESS_OVER_MAJORITY = 0.05


def target_label(question_type: str, row) -> str:
    """Return a comparable target label for the question type.

    Relative-depth answers are item-specific object names, so the meaningful
    language-only target is whether the answer is the first or second named
    option. Other question types use their released answer directly.
    """
    answer = str(row["answer"])
    if question_type != "relative_depth":
        return answer

    answer_space = str(row["answer_space"]).split("|")
    if len(answer_space) != 2 or answer not in answer_space:
        raise ValueError(
            "relative_depth answer must exactly match one of two answer_space values"
        )
    return "first" if answer == answer_space[0] else "second"


def evaluate_question_type(
    question_type: str,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    random_state: int,
    max_excess_over_majority: float = MAX_EXCESS_OVER_MAJORITY,
) -> dict:
    """Fit TF-IDF logistic regression and compare it with train majority."""
    if train_frame.empty or evaluation_frame.empty:
        raise ValueError(f"Cannot evaluate empty {question_type} train/evaluation data")

    train_labels = [target_label(question_type, row) for _, row in train_frame.iterrows()]
    evaluation_labels = [
        target_label(question_type, row) for _, row in evaluation_frame.iterrows()
    ]
    evaluation_majority_label = Counter(evaluation_labels).most_common(1)[0][0]
    majority_predictions = [evaluation_majority_label] * len(evaluation_labels)
    majority_baseline = float(accuracy_score(evaluation_labels, majority_predictions))

    if len(set(train_labels)) == 1:
        predictions = [train_labels[0]] * len(evaluation_labels)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        train_features = vectorizer.fit_transform(train_frame["question"].astype(str))
        evaluation_features = vectorizer.transform(evaluation_frame["question"].astype(str))
        classifier = LogisticRegression(max_iter=2000, random_state=random_state)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(evaluation_features)

    accuracy = float(accuracy_score(evaluation_labels, predictions))
    excess = accuracy - majority_baseline
    return {
        "question_type": question_type,
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "excess_over_majority": excess,
        "threshold": max_excess_over_majority,
        "passes": excess <= max_excess_over_majority + 1e-12,
        "train_items": len(train_frame),
        "evaluation_items": len(evaluation_frame),
    }
