"""Loads the sampled audit CSV and resolves each question's overlay evidence.

An audit row is expected to follow the CSV schema in
DATASET_CREATION_PLAN.md §11 (``question_id, image_id, question_type,
question, answer, evidence, ...``); extra columns are kept but ignored.

``evidence`` is documented in §10.2 as "a JSON list of the fact ids [the
question] relies on"; the rule-based generators (§4, P2) had not been written
when this tool was built, so the exact fact-id shape was not yet fixed. This
module accepts a JSON list of object indices (or of dicts carrying an
``object_index``/``index``/``object`` key), and falls back to matching the
scene's own object names against the question text when ``evidence`` is
missing, blank, or in some other shape — so the overlay degrades gracefully
instead of breaking if the eventual generator's evidence format differs.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tools.audit_app.scene_index import SceneIndex

REQUIRED_COLUMNS = ("question_id", "image_id", "question_type", "question", "answer")

# Which evidence keys name an object that is spoken aloud in the *question
# text itself* (as opposed to the answer, e.g. nearest_object's
# answer_concept, which must stay unbolded so the overlay doesn't spoil it).
# identify_superlative has no object in its question ("which object is
# largest?") so it is deliberately absent here.
HIGHLIGHT_EVIDENCE_KEYS: dict[str, tuple[str, ...]] = {
    "existence": ("concept",),
    "left_right": ("a_concept", "b_concept"),
    "relative_depth": ("a_concept", "b_concept"),
    "nearest_object": ("anchor_concept",),
}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6",
}


def canonicalize_answer(text: str) -> str:
    """Light-weight approximation of the Rule Q3 answer canonicaliser, used
    only to flag agreement between an annotator's own answer and gold in the
    audit UI. The dataset's real evaluation canonicaliser (§9) is shipped
    separately as `evaluate.py`; this is not a substitute for it."""
    normalized = str(text).strip().lower().rstrip(".!?")
    normalized = re.sub(r"\s+", " ", normalized).replace("_", " ")
    return _NUMBER_WORDS.get(normalized, normalized)


@dataclass(frozen=True)
class AuditItem:
    question_id: str
    image_id: str
    question_type: str
    question: str
    answer: str
    row: dict = field(repr=False)
    evidence_object_indices: tuple[int, ...] = ()
    highlight_words: tuple[str, ...] = ()

    @property
    def sensor(self) -> str | None:
        return self.row.get("sensor")

    @property
    def scene_type(self) -> str | None:
        return self.row.get("scene_type")

    def to_public_dict(self) -> dict:
        """Item fields sent to the client, gold answer included: the audit
        workflow shows gold immediately alongside the question rather than
        gating it behind a blind guess (see main.py's module docstring for
        the reasoning and the trade-off this accepts)."""
        return {
            "question_id": self.question_id,
            "image_id": self.image_id,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "sensor": self.sensor,
            "scene_type": self.scene_type,
            "evidence_object_indices": list(self.evidence_object_indices),
            "highlight_words": list(self.highlight_words),
        }


def _load_evidence_json(evidence_raw: object) -> object | None:
    if evidence_raw is None or (isinstance(evidence_raw, float) and math.isnan(evidence_raw)):
        return None
    text = str(evidence_raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_highlight_words(
    question_type: str, evidence_raw: object, concept_display_names: dict[str, str]
) -> tuple[str, ...]:
    """Display names of the objects named in `question` itself, in evidence
    order, for the audit UI to bold (readability only — never reveals the
    answer for types like nearest_object where the answer object is not
    among the named keys, see HIGHLIGHT_EVIDENCE_KEYS)."""
    keys = HIGHLIGHT_EVIDENCE_KEYS.get(question_type, ())
    if not keys:
        return ()
    parsed = _load_evidence_json(evidence_raw)
    if not isinstance(parsed, dict):
        return ()

    words: list[str] = []
    for key in keys:
        concept = parsed.get(key)
        if not isinstance(concept, str) or not concept:
            continue
        display_name = concept_display_names.get(concept, concept.replace("_", " "))
        if display_name not in words:
            words.append(display_name)
    return tuple(words)


def _parse_evidence_indices(evidence_raw: object) -> set[int] | None:
    parsed = _load_evidence_json(evidence_raw)
    if parsed is None:
        return None

    if isinstance(parsed, list):
        indices: set[int] = set()
        for entry in parsed:
            if isinstance(entry, int):
                indices.add(entry)
            elif isinstance(entry, dict):
                for key in ("object_index", "index", "object"):
                    value = entry.get(key)
                    if isinstance(value, int):
                        indices.add(value)
                        break
        return indices or None

    if isinstance(parsed, dict):
        # The P2 generators (dataset_creation/v2/*.py) settled on a per-type
        # dict shape rather than the list-of-fact-ids this module originally
        # anticipated (see this file's module docstring) — e.g. existence's
        # {"object_index": 3}, count's {"object_indices": [3, 7]},
        # identify_superlative's {"winner_object_index": 3, ...}. Every key
        # ending in one of those two names is pulled out; a type whose
        # evidence carries neither (relative_depth, left_right — concept
        # names and geometry only, no stored index) still resolves through
        # the name-matching fallback below, since both of those question
        # types name their objects directly in the question text.
        indices = set()
        for key, value in parsed.items():
            if key.endswith("object_index") and isinstance(value, int):
                indices.add(value)
            elif key.endswith("object_indices") and isinstance(value, list):
                indices.update(item for item in value if isinstance(item, int))
        return indices or None

    return None


def _object_names_mentioned_in(question: str, candidate_names: set[str]) -> set[str]:
    """Case-insensitive match of each candidate name against the question,
    tolerant of a trailing plural `s`/`es` (questions ask "how many lamps",
    not "lamp") — not a full singulariser, so an irregular plural like
    shelf/shelves still needs an explicit `evidence` value to be found."""
    question_lower = question.lower()
    mentioned = set()
    for name in candidate_names:
        if not name:
            continue
        pattern = r"\b" + re.escape(name.lower().replace("_", " ")) + r"(e?s)?\b"
        if re.search(pattern, question_lower):
            mentioned.add(name)
    return mentioned


def resolve_evidence_object_indices(
    question: str, evidence_raw: object, scene_index: SceneIndex, image_id: str
) -> tuple[int, ...]:
    parsed = _parse_evidence_indices(evidence_raw)
    if parsed is not None:
        return tuple(sorted(parsed))
    scene = scene_index.get(image_id)
    if scene is None:
        return ()
    mentioned = _object_names_mentioned_in(question, set(scene.object_names))
    return tuple(sorted(scene_index.object_indices_matching_names(image_id, mentioned)))


def load_audit_items(
    csv_path: Path, scene_index: SceneIndex, concept_display_names: dict[str, str] | None = None
) -> list[AuditItem]:
    frame = pd.read_csv(csv_path, dtype={"question_id": str, "image_id": str})
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")

    concept_display_names = concept_display_names or {}
    items: list[AuditItem] = []
    for row in frame.to_dict(orient="records"):
        question = str(row["question"])
        question_type = str(row["question_type"])
        evidence_raw = row.get("evidence")
        evidence = resolve_evidence_object_indices(
            question, evidence_raw, scene_index, str(row["image_id"])
        )
        highlight_words = extract_highlight_words(question_type, evidence_raw, concept_display_names)
        items.append(AuditItem(
            question_id=str(row["question_id"]),
            image_id=str(row["image_id"]),
            question_type=question_type,
            question=question,
            answer=str(row["answer"]),
            row=row,
            evidence_object_indices=evidence,
            highlight_words=highlight_words,
        ))
    return items
