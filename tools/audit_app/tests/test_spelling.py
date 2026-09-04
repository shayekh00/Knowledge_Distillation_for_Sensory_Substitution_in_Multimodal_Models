from tools.audit_app.spelling import candidate_answers_for, correct_spelling

CANONICAL_NAMES = ["chair", "table", "trash can", "night stand"]


def test_candidate_answers_for_closed_types_ignore_the_canonical_list():
    assert candidate_answers_for("existence", CANONICAL_NAMES) == ["yes", "no"]
    assert candidate_answers_for("left_right", CANONICAL_NAMES) == ["left", "right"]
    assert candidate_answers_for("count", CANONICAL_NAMES) == ["1", "2", "3", "4", "5"]


def test_candidate_answers_for_open_types_uses_the_canonical_list():
    assert candidate_answers_for("nearest_object", CANONICAL_NAMES) == CANONICAL_NAMES


def test_correct_spelling_exact_match_is_case_and_underscore_insensitive():
    assert correct_spelling("CHAIR", CANONICAL_NAMES) == "chair"
    assert correct_spelling("trash_can", CANONICAL_NAMES) == "trash can"


def test_correct_spelling_fixes_a_typo():
    assert correct_spelling("chiar", CANONICAL_NAMES) == "chair"
    assert correct_spelling("tabel", CANONICAL_NAMES) == "table"


def test_correct_spelling_number_words_map_to_digits():
    assert correct_spelling("five", ["1", "2", "3", "4", "5"]) == "5"
    assert correct_spelling("Three", ["1", "2", "3", "4", "5"]) == "3"


def test_correct_spelling_leaves_unrelated_text_unchanged():
    assert correct_spelling("refrigerator", CANONICAL_NAMES) == "refrigerator"


def test_correct_spelling_blank_input_stays_blank():
    assert correct_spelling("   ", CANONICAL_NAMES) == ""
