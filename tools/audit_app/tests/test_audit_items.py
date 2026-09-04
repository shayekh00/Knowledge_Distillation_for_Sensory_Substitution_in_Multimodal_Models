from pathlib import Path

from tools.audit_app.audit_items import resolve_evidence_object_indices
from tools.audit_app.scene_index import SceneIndex, SceneRecord


def _scene_index_with_one_scene(object_names: list[str]) -> SceneIndex:
    index = SceneIndex(Path("/does/not/exist.jsonl"), Path("."))
    index._records["scene/1"] = SceneRecord(
        image_id="scene/1", sensor="kv2", scene_type="bedroom",
        image_width=100, image_height=100,
        rgb_path=Path("rgb.jpg"), annotation_path=Path("does-not-exist.json"),
        object_names=object_names,
    )
    return index


def test_evidence_json_list_of_indices_is_used_directly():
    scene_index = _scene_index_with_one_scene(["bed", "lamp"])

    resolved = resolve_evidence_object_indices("irrelevant text", "[0]", scene_index, "scene/1")

    assert resolved == (0,)


def test_falls_back_to_name_matching_when_evidence_is_missing():
    scene_index = _scene_index_with_one_scene(["bed", "lamp"])

    resolved = resolve_evidence_object_indices(
        "How many lamps are in the image?", None, scene_index, "scene/1"
    )

    assert resolved == (1,)  # matches "lamp" despite the plural "lamps" in the question


def test_falls_back_to_empty_when_evidence_unparseable_and_nothing_mentioned():
    scene_index = _scene_index_with_one_scene(["bed", "lamp"])

    resolved = resolve_evidence_object_indices(
        "What kind of room is this?", "not json", scene_index, "scene/1"
    )

    assert resolved == ()


def test_evidence_dict_with_object_index_key_is_used_directly():
    # e.g. existence.py's {"concept": "lamp", "object_index": 1, "area_frac": 0.05}
    scene_index = _scene_index_with_one_scene(["bed", "lamp"])

    resolved = resolve_evidence_object_indices(
        "irrelevant text", '{"concept": "lamp", "object_index": 1}', scene_index, "scene/1"
    )

    assert resolved == (1,)


def test_evidence_dict_with_object_indices_list_key_is_used_directly():
    # e.g. count.py's {"concept": "chair", "count": 2, "object_indices": [0, 2]}
    scene_index = _scene_index_with_one_scene(["chair", "bed", "chair"])

    resolved = resolve_evidence_object_indices(
        "irrelevant text", '{"concept": "chair", "object_indices": [0, 2]}', scene_index, "scene/1"
    )

    assert resolved == (0, 2)


def test_evidence_dict_with_suffixed_object_index_key_is_used_directly():
    # e.g. identify_superlative.py's {"winner_object_index": 0, "runner_up_concept": "bed"}
    scene_index = _scene_index_with_one_scene(["lamp", "bed"])

    resolved = resolve_evidence_object_indices(
        "irrelevant text", '{"winner_object_index": 0, "runner_up_concept": "bed"}', scene_index, "scene/1"
    )

    assert resolved == (0,)


def test_evidence_dict_without_any_index_key_falls_back_to_name_matching():
    # e.g. relative_depth.py's {"a_concept": "bed", "b_concept": "lamp", ...} — no
    # stored index at all, so this must fall through to the text-matching path.
    scene_index = _scene_index_with_one_scene(["bed", "lamp"])

    resolved = resolve_evidence_object_indices(
        "Which is closer to the camera, the bed or the lamp?",
        '{"a_concept": "bed", "b_concept": "lamp", "comparative": "closer"}',
        scene_index, "scene/1",
    )

    assert resolved == (0, 1)
