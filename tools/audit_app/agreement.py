"""
Decides whether a model's free-form answer agrees with a gold answer.

This is deliberately the *dataset's own* canonicaliser (Rule V1 in
docs/DATASET_CREATION_PLAN.md — normalise, then the hand-written synonym
table) rather than fuzzy string matching. That distinction is the whole
ballgame for triage quality: "couch" must count as agreement with "sofa"
because synonyms.csv says they are the same concept, while a genuinely
different answer must not get snapped onto a similar-looking canonical
name just because the edit distance is small. Naive string comparison
would flag hundreds of false disagreements and waste the reviewer's time;
fuzzy matching would hide real ones.

The implementation lives in `dataset/dataset_creation/v2/answer_form.py` so
that `evaluate.py` scores predictions with exactly the function the audit
tool triages with (§13.6). This module stays as the audit app's entry point.
"""
from __future__ import annotations

import os
import sys

_V2_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset", "dataset_creation", "v2",
)
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from answer_form import (  # noqa: E402,F401  (re-exported for the audit app)
    FIXED_ANSWER_TYPES,
    answers_agree,
    canonical_answer_form,
)

__all__ = ["FIXED_ANSWER_TYPES", "answers_agree", "canonical_answer_form"]
