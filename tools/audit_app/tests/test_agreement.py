from tools.audit_app.agreement import answers_agree, canonical_answer_form

SYNONYMS = {"couch": "sofa", "tv": "television"}
VOCAB = {
    "sofa": {"display_name": "sofa", "category": "furniture", "is_structural": False},
    "chair": {"display_name": "chair", "category": "furniture", "is_structural": False},
    "television": {"display_name": "television", "category": "electronics", "is_structural": False},
    "nightstand": {"display_name": "night_stand", "category": "furniture", "is_structural": False},
}


def agree(model, gold, qtype="nearest_object"):
    return answers_agree(model, gold, qtype, SYNONYMS, VOCAB)


def test_case_and_article_differences_still_agree():
    assert agree("Bookshelf", "bookshelf")
    assert agree("The chair", "chair")


def test_synonym_table_is_honoured():
    # synonyms.csv says couch and sofa are the same concept, so a model
    # answering "couch" has not disagreed with gold "sofa".
    assert agree("couch", "sofa")
    assert agree("tv", "television")


def test_underscored_gold_matches_spaced_model_answer():
    assert agree("night stand", "night_stand")


def test_genuinely_different_answers_disagree():
    assert not agree("chair", "sofa")


def test_similar_looking_but_different_answers_are_not_snapped_together():
    # A fuzzy matcher might collapse these; the synonym table must not.
    assert not agree("chair", "television")


def test_number_words_agree_with_digits_for_count():
    assert agree("five", "5", qtype="count")
    assert agree("3", "3", qtype="count")
    assert not agree("2", "3", qtype="count")


def test_yes_no_and_left_right_compare_literally():
    assert agree("Yes.", "yes", qtype="existence")
    assert not agree("no", "yes", qtype="existence")
    assert agree("Left", "left", qtype="left_right")


def test_blank_model_answer_never_agrees():
    assert not agree("", "chair")
    assert not agree("   ", "chair")


def test_out_of_vocab_answers_compare_on_their_normalised_form():
    # "refrigerator" is not in this test vocab; it must still compare
    # equal to itself rather than being snapped onto something else.
    assert agree("refrigerator", "refrigerator")
    assert not agree("refrigerator", "chair")


def test_canonical_answer_form_is_stable_for_fixed_types():
    assert canonical_answer_form("YES", "existence", SYNONYMS, VOCAB) == "yes"
    assert canonical_answer_form("four", "count", SYNONYMS, VOCAB) == "4"
