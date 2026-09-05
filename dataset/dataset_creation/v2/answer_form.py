"""The dataset's single canonicaliser for answers — gold and predicted alike.

DATASET_CREATION_PLAN.md §13.6 requires one shared canonicaliser for gold and
predictions, so a model is never marked wrong for a surface form the dataset
itself would have accepted ("couch" for `sofa`, "Three." for `3`). Putting it
here rather than in either caller is what makes that guarantee checkable:
`evaluate.py` (§9) and the audit app's triage (`tools/audit_app/agreement.py`)
both import this module, so the two can never drift apart.

Deliberately not fuzzy matching. Synonyms resolve through the hand-written
table (Rule V1); anything else compares exactly. Edit-distance matching would
snap a genuinely different answer onto a similar-looking canonical name.
"""
from __future__ import annotations

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from vocab import canonicalize  # noqa: E402

# Types whose answers are a small fixed set of literals, not object names.
# Running "left" or "yes" through the object vocabulary would be meaningless.
FIXED_ANSWER_TYPES = {"existence", "left_right"}

_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_WHITESPACE_RE = re.compile(r"\s+")

# Rule Q3: numbers are digits. Spelled-out forms are accepted from a model and
# folded to the gold surface form. Small explicit table rather than a parser —
# gold counts never exceeded five even before `count` was retired (§13.16).
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}


def canonical_answer_form(text: str, question_type: str,
                          synonym_map: dict, canonical_vocab: dict) -> str:
    """Comparable form of an answer string, for `question_type`."""
    normalized = str(text or "").strip().lower().rstrip(".!?")
    normalized = _WHITESPACE_RE.sub(" ", normalized).replace("_", " ")
    normalized = _LEADING_ARTICLE_RE.sub("", normalized)
    normalized = _NUMBER_WORDS.get(normalized, normalized)
    if not normalized or question_type in FIXED_ANSWER_TYPES:
        return normalized
    return canonicalize(normalized, synonym_map, canonical_vocab)["display_name"].replace("_", " ")


def answers_agree(model_answer: str, gold_answer: str, question_type: str,
                  synonym_map: dict, canonical_vocab: dict) -> bool:
    """True when a predicted answer canonicalises to the same form as gold."""
    if not str(model_answer or "").strip():
        return False
    return (canonical_answer_form(model_answer, question_type, synonym_map, canonical_vocab)
            == canonical_answer_form(gold_answer, question_type, synonym_map, canonical_vocab))
