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
"""
from __future__ import annotations

import os
import re
import sys

_V2_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset", "dataset_creation", "v2",
)
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from vocab import canonicalize  # noqa: E402  (dataset_creation/v2/vocab.py)

# Types whose answers are a small fixed set of literals, not object names —
# these must not be run through the object vocabulary.
FIXED_ANSWER_TYPES = {"existence", "left_right", "count"}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def canonical_answer_form(text: str, question_type: str,
                           synonym_map: dict, canonical_vocab: dict) -> str:
    """Comparable form of an answer string, for `question_type`."""
    normalized = str(text or "").strip().lower().rstrip(".!?")
    normalized = _WHITESPACE_RE.sub(" ", normalized).replace("_", " ")
    normalized = _LEADING_ARTICLE_RE.sub("", normalized)  # "the bookshelf" -> "bookshelf"
    normalized = _NUMBER_WORDS.get(normalized, normalized)
    if not normalized or question_type in FIXED_ANSWER_TYPES:
        return normalized
    resolved = canonicalize(normalized, synonym_map, canonical_vocab)
    return resolved["display_name"].replace("_", " ")


def answers_agree(model_answer: str, gold_answer: str, question_type: str,
                   synonym_map: dict, canonical_vocab: dict) -> bool:
    if not str(model_answer or "").strip():
        return False
    model_form = canonical_answer_form(model_answer, question_type, synonym_map, canonical_vocab)
    gold_form = canonical_answer_form(gold_answer, question_type, synonym_map, canonical_vocab)
    return model_form == gold_form
