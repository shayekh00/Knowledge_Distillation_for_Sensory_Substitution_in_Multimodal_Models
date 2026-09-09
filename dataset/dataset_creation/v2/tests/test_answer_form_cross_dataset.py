"""answer_form.py's Phase 4 audit (arkitscenes_plan.md §6): unlike
build_vocab.py/generator_common.py/build_release.py/freeze_release.py, this
module takes `synonym_map`/`canonical_vocab` as plain parameters rather than
loading them from a hardcoded path, so it needs no --dataset flag — it is
already dataset-agnostic by construction. This pins that down against the
real ARKitScenes vocab rather than leaving it as an unverified reading, and
locks in the two vocab fixes build_vocab.py needed (`tv_monitor` ->
`television`, `washer`'s category) as visible through this module too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from answer_form import answers_agree, canonical_answer_form  # noqa: E402
from vocab import load_canonical_vocab, load_synonyms  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
SYNONYM_MAP = load_synonyms(os.path.join(REPO_ROOT, "data", "vocab", "synonyms.csv"))
ARKIT_VOCAB = load_canonical_vocab(
    os.path.join(REPO_ROOT, "data", "vocab_arkit", "canonical_objects.csv"))


def test_arkitscenes_target_class_answers_canonicalise_to_themselves():
    for concept in ("cabinet", "washer", "television", "dishwasher"):
        assert canonical_answer_form(
            concept, "identify_superlative", SYNONYM_MAP, ARKIT_VOCAB) == concept


def test_tv_monitor_synonym_agrees_with_television_under_the_arkit_vocab():
    assert answers_agree("tv", "television", "identify_superlative", SYNONYM_MAP, ARKIT_VOCAB)


def test_couch_synonym_agrees_with_sofa_under_the_arkit_vocab():
    assert answers_agree("couch", "sofa", "nearest_object", SYNONYM_MAP, ARKIT_VOCAB)


def test_fixed_answer_types_bypass_the_vocab_entirely():
    assert canonical_answer_form("Left.", "left_right", SYNONYM_MAP, ARKIT_VOCAB) == "left"
    assert canonical_answer_form("Yes", "existence", SYNONYM_MAP, ARKIT_VOCAB) == "yes"
