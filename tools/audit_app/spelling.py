"""
Lightweight spelling correction for the audit tool's "your answer" box
(used by main.py when an annotator marks an item "incorrect" and types
what they think the right answer is).

Correction targets are always the dataset's own controlled answer
vocabulary — yes/no, left/right, or the canonical object names in
data/vocab/canonical_objects.csv — never a general
English dictionary. That distinction matters: v1's neural spell-corrector
silently mutated gold labels toward whatever English word was statistically
likely ("red"->"bed", see docs/DATASET_CREATION_PLAN.md D7). Here the
annotator's typed text can only be corrected *toward* an answer that is
already valid for that question type, and if nothing is close enough it
is left alone rather than forced onto the wrong candidate.
"""
from __future__ import annotations

import difflib

FIXED_ANSWER_SPACES = {
    "existence": ("yes", "no"),
    "left_right": ("left", "right"),
}


def candidate_answers_for(question_type: str, canonical_display_names: list) -> list:
    """The fixed answer space for closed types; the full canonical object
    vocabulary for the open types (identify_superlative, nearest_object,
    relative_depth)."""
    fixed = FIXED_ANSWER_SPACES.get(question_type)
    return list(fixed) if fixed is not None else list(canonical_display_names)


def correct_spelling(raw_answer: str, candidates: list, cutoff: float = 0.6) -> str:
    """Best-matching member of `candidates` for `raw_answer`, or
    `raw_answer` unchanged (just trimmed) if nothing is close enough.
    Exact match (case/underscore-insensitive) is tried before fuzzy
    matching."""
    normalized = raw_answer.strip()
    if not normalized:
        return normalized

    lowered = normalized.lower().replace("_", " ")

    candidate_lookup = {candidate.lower().replace("_", " "): candidate for candidate in candidates}
    if lowered in candidate_lookup:
        return candidate_lookup[lowered]

    close_matches = difflib.get_close_matches(lowered, candidate_lookup.keys(), n=1, cutoff=cutoff)
    return candidate_lookup[close_matches[0]] if close_matches else normalized
