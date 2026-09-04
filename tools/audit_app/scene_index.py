"""Read-only access to the P0 scene index and raw SUNRGBD annotation polygons.

Both are the outputs of ``dataset/dataset_creation/v2/build_index.py`` and the
original SUNRGBD release; this module never writes to them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenePolygon:
    object_index: int
    name: str
    x: list[float]
    y: list[float]


@dataclass(frozen=True)
class SceneRecord:
    image_id: str
    sensor: str
    scene_type: str
    image_width: int
    image_height: int
    rgb_path: Path
    annotation_path: Path
    object_names: list[str]


class SceneIndex:
    """Loads ``data/index/scene_index.jsonl`` once and serves lookups by image_id."""

    def __init__(self, index_path: Path, dataset_root: Path) -> None:
        self._dataset_root = dataset_root
        self._records: dict[str, SceneRecord] = {}
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as index_file:
                for line in index_file:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._records[row["image_id"]] = SceneRecord(
                        image_id=row["image_id"],
                        sensor=row["sensor"],
                        scene_type=row["scene_type"],
                        image_width=row["image_width"],
                        image_height=row["image_height"],
                        rgb_path=dataset_root / row["rgb_path"],
                        annotation_path=dataset_root / row["annotation_path"],
                        object_names=[obj["raw_name"] for obj in row["objects"]],
                    )

    def get(self, image_id: str) -> SceneRecord | None:
        return self._records.get(image_id)

    def __len__(self) -> int:
        return len(self._records)

    def polygons_for(self, image_id: str, object_indices: set[int] | None = None) -> list[ScenePolygon]:
        """Polygons for `image_id`, optionally restricted to `object_indices`.

        Reads the raw SUNRGBD ``annotation/index.json`` directly rather than
        the scene index (which stores only area/centroid/depth summaries, not
        vertex coordinates). Returns [] if the annotation is missing or
        unparseable rather than raising, since a failed overlay must never
        block reviewing the underlying question.
        """
        record = self.get(image_id)
        if record is None or not record.annotation_path.is_file():
            return []
        try:
            annotation = json.loads(record.annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        object_names = [
            obj.get("name", "") if isinstance(obj, dict) else ""
            for obj in annotation.get("objects", [])
        ]
        frames = annotation.get("frames", [])
        if not frames:
            return []

        polygons: list[ScenePolygon] = []
        for entry in frames[0].get("polygon", []):
            object_index = entry.get("object")
            if object_index is None:
                continue
            if object_indices is not None and object_index not in object_indices:
                continue
            xs, ys = entry.get("x", []), entry.get("y", [])
            xs = xs if isinstance(xs, list) else [xs]
            ys = ys if isinstance(ys, list) else [ys]
            if len(xs) < 3:
                continue
            name = object_names[object_index] if object_index < len(object_names) else ""
            polygons.append(ScenePolygon(object_index=object_index, name=name, x=xs, y=ys))
        return polygons

    def object_indices_matching_names(self, image_id: str, mentioned_names: set[str]) -> set[int]:
        """Fallback evidence resolver: object indices whose raw name (case-
        insensitive, underscores/spaces interchangeable) is one of
        `mentioned_names`. Used when a question row carries no usable
        `evidence` column — see ``audit_items.resolve_evidence_object_indices``.
        """
        record = self.get(image_id)
        if record is None:
            return set()
        normalized_targets = {name.lower().replace("_", " ").strip() for name in mentioned_names}
        return {
            index
            for index, raw_name in enumerate(record.object_names)
            if raw_name.lower().replace("_", " ").strip() in normalized_targets
        }
