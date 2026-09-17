from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from m3fd_common import (capture_group, canonicalize_class, clip_box, map_box_between_frames,
                         paired_keys, parse_voc_xml)
import build_index_m3fd


def test_voc_parse_canonicalizes_published_classes_and_reason_codes_unknown(tmp_path):
    annotation = tmp_path / "frame.xml"
    annotation.write_text("""<annotation><size><width>100</width><height>50</height></size>
    <object><name>Pedestrian</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object>
    <object><name>bicycle</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object>
    <object><name>car</name><bndbox><xmin>5</xmin><ymin>2</ymin><xmax>5</xmax><ymax>8</ymax></bndbox></object>
    </annotation>""", encoding="utf-8")
    size, objects, rejected = parse_voc_xml(annotation)
    assert size == (100, 50)
    assert objects == [{"object_index": 0, "raw_name": "Pedestrian", "concept": "person", "box_xyxy": [1.0, 2.0, 20.0, 30.0]}]
    assert [row["reason_code"] for row in rejected] == ["UNKNOWN_CLASS", "INVALID_BOX"]


def test_mapping_clipping_and_capture_group_are_explicit():
    assert map_box_between_frames([10, 5, 20, 25], (100, 50), (200, 200)) == [20.0, 20.0, 40.0, 100.0]
    assert clip_box([-5, 2, 10, 40], 20, 20) == [0.0, 2, 10, 20.0]
    assert clip_box([30, 2, 40, 6], 20, 20) is None
    assert capture_group("day/sequence_002_frame_001", r"(sequence_\d+)_frame") == "sequence_002"
    assert capture_group("frame_001", r"(sequence_\d+)") is None


def test_pairing_reports_every_missing_partner():
    rgb = {"a": Path("a.jpg"), "b": Path("b.jpg")}
    thermal = {"a": Path("a.png"), "c": Path("c.png")}
    annotations = {"a": Path("a.xml"), "b": Path("b.xml"), "d": Path("d.xml")}
    complete, report = paired_keys(rgb, thermal, annotations)
    assert complete == ["a"]
    assert report["rgb_without_thermal"] == ["b"]
    assert report["rgb_without_annotation"] == []
    assert report["thermal_without_rgb"] == ["c"]
    assert report["annotation_without_rgb"] == ["d"]


def test_invalid_xml_is_not_silently_accepted(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text("<annotation>", encoding="utf-8")
    with pytest.raises(ValueError, match="unparseable XML"):
        parse_voc_xml(path)
    assert canonicalize_class("motor-bike") == "motorcycle"


def test_index_record_maps_rgb_boxes_to_thermal_and_rejects_unreviewed(tmp_path, monkeypatch):
    """The index keeps both source and thermal-frame evidence, never depth_path."""
    monkeypatch.setattr(build_index_m3fd, "DATASET_ROOT", tmp_path)
    rgb_path, thermal_path, annotation_path = tmp_path / "rgb.jpg", tmp_path / "thermal.png", tmp_path / "frame.xml"
    Image.new("RGB", (100, 50)).save(rgb_path)
    Image.new("L", (200, 100)).save(thermal_path)
    annotation_path.write_text("""<annotation><size><width>100</width><height>50</height></size>
    <object><name>car</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>20</xmax><ymax>25</ymax></bndbox></object>
    </annotation>""", encoding="utf-8")
    drops = []
    record = build_index_m3fd.build_record(
        "session_01/frame_0001", rgb_path, thermal_path, annotation_path,
        r"(session_\d+)/", "rgb", True, {"session_01/frame_0001": True}, True, drops,
    )
    assert drops == []
    assert record["rgb_path"] == "rgb.jpg"
    assert record["thermal_path"] == "thermal.png"
    assert "depth_path" not in record
    assert record["objects"][0]["original_box_xyxy"] == [10.0, 5.0, 20.0, 25.0]
    assert record["objects"][0]["thermal_box_xyxy"] == [20.0, 10.0, 40.0, 50.0]
    rejected = build_index_m3fd.build_record(
        "session_02/frame_0001", rgb_path, thermal_path, annotation_path,
        r"(session_\d+)/", "rgb", True, {}, True, drops,
    )
    assert rejected is None
    assert drops[-1]["reason_code"] == "REGISTRATION_UNREVIEWED"
