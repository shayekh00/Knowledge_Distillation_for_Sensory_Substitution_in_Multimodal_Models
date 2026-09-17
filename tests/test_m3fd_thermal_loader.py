from __future__ import annotations

import csv
import os
import sys

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.dataloader.M3FDThermalDataset import M3FDThermalDataset
from distillation.inference_isolation import FileAccessTracer


def test_thermal_loader_never_opens_rgb_or_annotation_paths(tmp_path):
    thermal_dir = tmp_path / "Infrared"; thermal_dir.mkdir()
    rgb_dir = tmp_path / "Visible"; rgb_dir.mkdir()
    annotation_dir = tmp_path / "Annotation"; annotation_dir.mkdir()
    Image.new("L", (8, 8), 12).save(thermal_dir / "frame.png")
    # Deliberately nonexistent RGB/annotation names: an accidental access fails.
    release = tmp_path / "test.csv"
    with release.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "question", "answer", "thermal_path", "rgb_path", "annotation_path"])
        writer.writeheader(); writer.writerow({"question_id": "x", "question": "Is there a car?", "answer": "yes", "thermal_path": "Infrared/frame.png", "rgb_path": "Visible/nope.jpg", "annotation_path": "Annotation/nope.xml"})
    dataset = M3FDThermalDataset(tmp_path, release)
    with FileAccessTracer() as trace: sample = dataset[0]
    assert sample["image"].size == (8, 8)
    trace.assert_untouched(["Visible", "Annotation"])
