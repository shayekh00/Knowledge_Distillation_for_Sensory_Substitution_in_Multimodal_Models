"""
Shared object-name normalization for VQA-SUNRGBD-v2 (Rule V1 in the plan).

normalize_raw_name() is the single function used everywhere a raw SUNRGBD
annotation name needs to become a canonical, answerable object name — the
vocab builder (P1) and every question generator (P2) import it from here so
gold answers and evaluation predictions are canonicalized identically.

Normalization order (fixed, do not reorder):
  1. lowercase, strip, collapse internal whitespace
  2. strip a trailing run of digits/underscores/hyphens ("wall23" -> "wall")
  3. singularize (inflect), with two guards against known inflect bugs:
     - words ending in "ss" are never singularized: inflect's singular_noun
       treats "glass"/"mattress" as plurals of "glas"/"mattres", which is
       wrong (the true plural of a double-s noun ends in "-sses").
     - a short exception list of plural-only nouns ("clothes", "scissors", ...)
       that have no singular form as a physical object.
  4. look up the result in the synonym table (data/vocab/synonyms.csv)
  5. look up the result in the canonical vocabulary
     (data/vocab/canonical_objects.csv); if absent, the name is returned
     with in_vocab=False (Rule V1 step 5 / Rule V4).
"""
from __future__ import annotations

import csv
import os
import re

import inflect

_INFLECT = inflect.engine()

_PLURAL_ONLY_NOUNS = {"clothes", "scissors", "pants", "pajamas", "shorts", "goggles"}
# "glasses" is deliberately not in this set: in an indoor-room dataset it is
# almost always the plural of a drinking "glass", not eyewear, so it should
# singularize normally.

_TRAILING_JUNK_RE = re.compile(r"[\d_\-]+$")
_WHITESPACE_RE = re.compile(r"\s+")


def _safe_singular(word: str) -> str:
    if word in _PLURAL_ONLY_NOUNS or word.endswith("ss"):
        return word
    candidate = _INFLECT.singular_noun(word)
    return candidate if candidate else word


def normalize_raw_name(raw_name: str | None) -> str:
    """Steps 1-3 only (no synonym/vocab lookup). Never raises."""
    name = (raw_name or "").strip().lower()
    name = _WHITESPACE_RE.sub(" ", name)
    name = _TRAILING_JUNK_RE.sub("", name).strip()
    if not name:
        return name
    return _safe_singular(name)


def load_synonyms(synonyms_csv_path: str) -> dict:
    """raw_normalized -> canonical_concept, per data/vocab/synonyms.csv."""
    synonym_map = {}
    with open(synonyms_csv_path, "r", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["raw_normalized"].startswith("#"):
                continue
            synonym_map[row["raw_normalized"]] = row["canonical_concept"]
    return synonym_map


def load_canonical_vocab(canonical_objects_csv_path: str) -> dict:
    """canonical_concept -> {display_name, category, is_structural}."""
    vocab = {}
    with open(canonical_objects_csv_path, "r", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            vocab[row["canonical_concept"]] = {
                "display_name": row["display_name"],
                "category": row["category"],
                "is_structural": row["is_structural"] == "True",
            }
    return vocab


def canonicalize(raw_name: str, synonym_map: dict, canonical_vocab: dict) -> dict:
    """
    Full Rule V1 pipeline. Returns a dict with:
      normalized, concept, in_vocab, display_name, category, is_structural
    """
    normalized = normalize_raw_name(raw_name)
    concept = synonym_map.get(normalized, normalized)
    entry = canonical_vocab.get(concept)
    if entry is None:
        return {
            "normalized": normalized,
            "concept": concept,
            "in_vocab": False,
            "display_name": concept.replace("_", " "),
            "category": None,
            "is_structural": False,
        }
    return {
        "normalized": normalized,
        "concept": concept,
        "in_vocab": True,
        "display_name": entry["display_name"],
        "category": entry["category"],
        "is_structural": entry["is_structural"],
    }
