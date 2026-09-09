"""SceneIndex.polygons_for's ARKitScenes path (arkitscenes_plan.md §6 Phase
4): ARKitScenes' raw annotation JSON has no SUN-RGB-D-shaped
`frames[0]["polygon"]` to parse, so the audit app's evidence overlays would
otherwise silently come back empty for every ARKitScenes item. When a scene
record carries `object_polygons_xy` (set only for ARKitScenes records),
`polygons_for` must use it directly instead of attempting the raw-
annotation parse — and must look up names/polygons by each object's own
`object_index`, not its position in the (possibly gap-having, since
ARKitScenes drops invisible objects) `object_names` list.
"""
from pathlib import Path

from tools.audit_app.scene_index import SceneIndex, SceneRecord


def _scene_index_with(object_names_by_index: dict, object_polygons_xy: dict) -> SceneIndex:
    index = SceneIndex(Path("/does/not/exist.jsonl"), Path("."))
    index._records["scene/1"] = SceneRecord(
        image_id="scene/1", sensor="arkit", scene_type="unknown",
        image_width=256, image_height=192,
        rgb_path=Path("rgb.jpg"), annotation_path=Path("does-not-exist.json"),
        object_names=list(object_names_by_index.values()),
        object_names_by_index=object_names_by_index,
        object_polygons_xy=object_polygons_xy,
    )
    return index


def test_uses_stored_polygon_xy_instead_of_parsing_missing_raw_annotation():
    scene_index = _scene_index_with(
        object_names_by_index={0: "chair", 2: "table"},
        object_polygons_xy={0: [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
                            2: [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]]},
    )

    polygons = scene_index.polygons_for("scene/1")

    assert {p.object_index for p in polygons} == {0, 2}
    chair = next(p for p in polygons if p.object_index == 0)
    assert chair.name == "chair"
    assert chair.x == [1.0, 2.0, 2.0]
    assert chair.y == [1.0, 1.0, 2.0]


def test_object_indices_filter_is_respected_with_gapped_indices():
    scene_index = _scene_index_with(
        object_names_by_index={0: "chair", 2: "table"},
        object_polygons_xy={0: [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
                            2: [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]]},
    )

    polygons = scene_index.polygons_for("scene/1", object_indices={2})

    assert [p.object_index for p in polygons] == [2]
    assert polygons[0].name == "table"


def test_name_lookup_uses_true_object_index_not_list_position():
    """Object 0 was dropped (not visible in this frame); object 3 is the
    only survivor. A position-based lookup would wrongly index into an
    empty/short list — this must resolve by the real index, 3."""
    scene_index = _scene_index_with(
        object_names_by_index={3: "sofa"},
        object_polygons_xy={3: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
    )

    polygons = scene_index.polygons_for("scene/1")

    assert len(polygons) == 1
    assert polygons[0].object_index == 3
    assert polygons[0].name == "sofa"


def test_falls_back_to_raw_annotation_when_no_polygon_xy_is_stored():
    """SUN-RGB-D records carry no polygon_xy at all — must not take the
    ARKitScenes branch, and must return [] (not raise) when the raw
    annotation file genuinely doesn't exist, exactly as before this change."""
    scene_index = _scene_index_with(
        object_names_by_index={0: "bed"},
        object_polygons_xy={},
    )

    assert scene_index.polygons_for("scene/1") == []
