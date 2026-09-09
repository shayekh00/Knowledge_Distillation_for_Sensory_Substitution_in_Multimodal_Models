"""cap_with_floor (balance.py): the ARKitScenes small-corpus release fix
(arkitscenes_plan.md §6 Phase 3, 2026-09-08). SUN-RGB-D's answer-balance
caps can legitimately trim a tiny pool to zero rows, which — via
build_release.py's shared-minimum split size (Rule 6.4) — silently emptied
every other question type's val/test out along with it. cap_with_floor
falls back to the uncapped pool exactly when the cap would drop below
min_keep, and is a no-op at min_keep=0 (every SUN-RGB-D call site).
"""
import pandas as pd

import balance


def test_falls_back_to_original_when_capped_result_is_too_small():
    original = pd.DataFrame({"answer": ["a"] * 14})
    capped = original.iloc[0:0]  # e.g. cap_answer_share_per_group dropped the whole group

    result = balance.cap_with_floor(capped, original, min_keep=1)

    assert len(result) == len(original)


def test_keeps_the_capped_result_when_it_already_clears_the_floor():
    original = pd.DataFrame({"answer": ["a"] * 10 + ["b"] * 10})
    capped = original.iloc[:5]

    result = balance.cap_with_floor(capped, original, min_keep=1)

    assert len(result) == 5


def test_min_keep_zero_is_a_no_op_regardless_of_size():
    original = pd.DataFrame({"answer": ["a"] * 10})
    capped = original.iloc[0:0]

    result = balance.cap_with_floor(capped, original, min_keep=0)

    assert len(result) == 0
