"""The M³FD-specific half of the audit app: evidence that names concepts and
boxes instead of object indices, and geometry indexed in the thermal frame.

The fixtures below mirror `m3fd_candidates.generate`'s real evidence dicts and
`build_index_m3fd.build_record`'s real index rows, so a change to either shape
breaks here rather than silently emptying the reviewer's overlay.
"""
import json
from pathlib import Path

from tools.audit_app.audit_items import (
    extract_highlight_words,
    resolve_evidence_object_indices,
)
from tools.audit_app.scene_index import SceneIndex

# One 640x512 thermal frame whose RGB partner is 1024x768 — M³FD's two frames
# are registered but not the same size, which is the whole reason the overlay
# has to be rescaled per modality.
THERMAL_SIZE = (640, 512)
RGB_SIZE = (1024, 768)
PERSON_BOX = [100.0, 100.0, 140.0, 200.0]
CAR_BOX = [400.0, 150.0, 560.0, 260.0]
SECOND_PERSON_BOX = [200.0, 110.0, 236.0, 205.0]


def _write_m3fd_index(tmp_path: Path) -> SceneIndex:
    boxes = [("person", PERSON_BOX), ("car", CAR_BOX), ("person", SECOND_PERSON_BOX)]
    record = {
        "image_id": "m3fd/00042",
        "sequence_id": "street_a",
        "sensor": "thermal",
        "scene_type": "unknown",
        "image_width": THERMAL_SIZE[0],
        "image_height": THERMAL_SIZE[1],
        "rgb_width": RGB_SIZE[0],
        "rgb_height": RGB_SIZE[1],
        "rgb_path": "M3FD/Visible/00042.png",
        "thermal_path": "M3FD/Infrared/00042.png",
        "annotation_path": "M3FD/Annotation/00042.xml",
        "objects": [
            {
                "object_index": index,
                # The source's raw name differs from the canonical concept,
                # which is exactly why concept matching is not name matching.
                "raw_name": {"person": "People", "car": "Car"}[concept],
                "concept": concept,
                "thermal_box_xyxy": box,
                "polygon_xy": [[box[0], box[1]], [box[2], box[1]],
                               [box[2], box[3]], [box[0], box[3]]],
            }
            for index, (concept, box) in enumerate(boxes)
        ],
    }
    index_path = tmp_path / "scene_index_m3fd.jsonl"
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return SceneIndex(index_path, tmp_path / "dataset")


def _resolve(scene_index: SceneIndex, question_type: str, evidence: dict) -> tuple[int, ...]:
    return resolve_evidence_object_indices(
        "irrelevant — M3FD never falls back to question text",
        json.dumps(evidence), scene_index, "m3fd/00042",
        question_type=question_type, evidence_style="m3fd",
    )


# ── Index loading ────────────────────────────────────────────────────────

def test_index_carries_both_modalities_and_their_own_frame_sizes(tmp_path):
    scene = _write_m3fd_index(tmp_path).get("m3fd/00042")

    assert scene.image_path("thermal").name == "00042.png"
    assert scene.image_path("thermal").parent.name == "Infrared"
    assert scene.image_path("rgb").parent.name == "Visible"
    # Indexed geometry is thermal-frame; the RGB frame reports its own size.
    assert scene.frame_size_for("thermal") == THERMAL_SIZE
    assert scene.frame_size_for("rgb") == RGB_SIZE


def test_single_modality_index_still_loads_with_no_thermal(tmp_path):
    record = {
        "image_id": "scene/1", "sensor": "kv2", "scene_type": "bedroom",
        "image_width": 640, "image_height": 480,
        "rgb_path": "SUNRGBD/a/image.jpg", "annotation_path": "SUNRGBD/a/index.json",
        "objects": [{"object_index": 0, "raw_name": "bed"}],
    }
    index_path = tmp_path / "scene_index.jsonl"
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    scene = SceneIndex(index_path, tmp_path / "dataset").get("scene/1")

    assert scene.thermal_path is None
    assert scene.image_path("thermal") is None
    assert scene.frame_size_for("rgb") == (640, 480)


def test_polygons_are_labelled_with_the_canonical_concept(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    names = {polygon.name for polygon in scene_index.polygons_for("m3fd/00042")}

    # "person", the word the question and gold answer use — not "People".
    assert names == {"person", "car"}


# ── Evidence resolution ──────────────────────────────────────────────────

def test_existence_resolves_every_box_it_recorded(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = _resolve(scene_index, "existence", {
        "rule": "complete_annotation_presence", "concept": "person", "count": 2,
        "boxes": [PERSON_BOX, SECOND_PERSON_BOX],
    })

    assert resolved == (0, 2)


def test_negative_existence_outlines_nothing(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = _resolve(scene_index, "existence", {
        "rule": "complete_annotation_presence", "concept": "bus", "count": 0, "boxes": [],
    })

    # Gold is "no"; there is no box to show, and inventing one would be worse
    # than showing none.
    assert resolved == ()


def test_count_falls_back_to_concept_because_it_records_no_boxes(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = _resolve(scene_index, "count", {
        "rule": "complete_annotation_count", "concept": "person", "count": 2,
    })

    # Both people, so the reviewer can actually recount them.
    assert resolved == (0, 2)


def test_left_right_resolves_its_two_boxes(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = _resolve(scene_index, "left_right", {
        "rule": "centroid_gap_and_iou", "a": "car", "b": "person",
        "a_box": CAR_BOX, "b_box": PERSON_BOX,
        "iou": 0.0, "horizontal_gap_px": 360.0, "minimum_gap_px": 64.0,
    })

    assert resolved == (0, 1)


def test_identify_superlative_outlines_winner_and_runner_up(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = _resolve(scene_index, "identify_superlative", {
        "rule": "largest_box_area_1.20_margin", "winner": "car",
        "winner_box": CAR_BOX, "winner_area_px": 17600.0,
        "runner_up_box": PERSON_BOX, "runner_up_area_px": 4000.0,
    })

    # The 1.20x margin is a claim about these two boxes; showing one alone
    # would make it uncheckable.
    assert resolved == (0, 1)


def test_box_matching_tolerates_the_csv_float_round_trip(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)
    nudged = [value + 0.2 for value in CAR_BOX]

    resolved = _resolve(scene_index, "left_right", {"a": "car", "b": "person", "a_box": nudged})

    assert resolved == (1,)


def test_a_box_matching_nothing_is_dropped_not_guessed(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = _resolve(scene_index, "left_right", {
        "a": "car", "b": "person", "a_box": CAR_BOX, "b_box": [1.0, 2.0, 3.0, 4.0],
    })

    assert resolved == (1,)


def test_unparseable_evidence_yields_no_overlay(tmp_path):
    scene_index = _write_m3fd_index(tmp_path)

    resolved = resolve_evidence_object_indices(
        "Is there a person?", "not json", scene_index, "m3fd/00042",
        question_type="existence", evidence_style="m3fd",
    )

    assert resolved == ()


# ── Highlighting ─────────────────────────────────────────────────────────

def test_left_right_highlights_its_short_keys(tmp_path):
    words = extract_highlight_words(
        "left_right", json.dumps({"a": "car", "b": "person"}), {}, evidence_style="m3fd")

    # SUN-RGB-D's table looks for a_concept/b_concept and would find nothing.
    assert words == ("car", "person")


def test_count_highlights_its_object_for_m3fd():
    words = extract_highlight_words(
        "count", json.dumps({"concept": "person", "count": 2}), {}, evidence_style="m3fd")

    assert words == ("person",)


def test_identify_superlative_never_highlights_its_winner():
    words = extract_highlight_words(
        "identify_superlative", json.dumps({"winner": "car"}), {}, evidence_style="m3fd")

    # `winner` is the gold answer; bolding it in the question would spoil it.
    assert words == ()
