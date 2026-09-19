"""Read-only access to the P0 scene index and raw SUNRGBD annotation polygons.

Both are the outputs of ``dataset/dataset_creation/v2/build_index.py`` and the
original SUNRGBD release; this module never writes to them.

ARKitScenes records carry their own projected-hull polygon per object
(``polygon_xy``, ``build_index_arkit.py`` — the same field
``left_right.py``'s IoU gate uses instead of re-deriving one) directly in
the scene index, since its raw ``*_3dod_annotation.json`` has no SUN-RGB-D-
shaped ``frames[0]["polygon"]`` to read (arkitscenes_plan.md §6 Phase 4).
``polygons_for`` prefers that when present rather than attempting the
SUN-RGB-D-specific raw-annotation parse.

M³FD records (``build_index_m3fd.py``) go one step further: they carry a
second registered image (``thermal_path``) alongside ``rgb_path``, a
``concept`` per object, and their geometry — ``image_width``/``image_height``,
``polygon_xy``, ``thermal_box_xyxy`` — in the **thermal** frame, with the RGB
frame's own size recorded separately as ``rgb_width``/``rgb_height``. The extra
fields below are all optional, so a SUN-RGB-D or ARKitScenes index loads
exactly as before.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    # Both fields below are keyed by each object's own `object_index` (not
    # list position): unlike SUN-RGB-D, ARKitScenes' indexer drops objects
    # that fail the visibility/occlusion gates, so a kept object's position
    # in `object_names` above is not reliably its original index — that
    # list only works by position because SUN-RGB-D never drops a mid-list
    # object. New code should prefer `object_names_by_index`.
    object_names_by_index: dict[int, str]
    object_polygons_xy: dict[int, list[list[float]]]
    # M³FD only; empty/None for every single-modality source (see module
    # docstring). `object_concepts_by_index` is the canonical class the
    # indexer assigned each box, which is what M³FD evidence names instead of
    # an object index; `object_boxes_by_index` is that box in the thermal
    # frame, which is how a box-valued evidence entry is matched back to it.
    thermal_path: Path | None = None
    rgb_width: int | None = None
    rgb_height: int | None = None
    object_concepts_by_index: dict[int, str] = field(default_factory=dict)
    object_boxes_by_index: dict[int, list[float]] = field(default_factory=dict)

    def image_path(self, modality: str) -> Path | None:
        """The file for one display modality, or None if this record has none.

        `image_width`/`image_height` describe the *default* modality's frame
        (thermal for M³FD, RGB everywhere else) — use `frame_size_for` rather
        than assuming they apply to whichever image is on screen.
        """
        if modality == "thermal":
            return self.thermal_path
        return self.rgb_path

    def frame_size_for(self, modality: str) -> tuple[int, int]:
        """Pixel dimensions of `modality`'s own frame.

        The indexed geometry lives in the default modality's frame, so drawing
        it over the other one means rescaling by the ratio of these two sizes.
        """
        if modality == "rgb" and self.rgb_width and self.rgb_height:
            return self.rgb_width, self.rgb_height
        return self.image_width, self.image_height


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
                        object_names_by_index={
                            obj["object_index"]: obj["raw_name"] for obj in row["objects"]
                        },
                        object_polygons_xy={
                            obj["object_index"]: obj["polygon_xy"]
                            for obj in row["objects"] if obj.get("polygon_xy")
                        },
                        thermal_path=(
                            dataset_root / row["thermal_path"] if row.get("thermal_path") else None
                        ),
                        rgb_width=row.get("rgb_width"),
                        rgb_height=row.get("rgb_height"),
                        object_concepts_by_index={
                            obj["object_index"]: obj["concept"]
                            for obj in row["objects"] if obj.get("concept")
                        },
                        object_boxes_by_index={
                            obj["object_index"]: obj["thermal_box_xyxy"]
                            for obj in row["objects"] if obj.get("thermal_box_xyxy")
                        },
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
        if record is None:
            return []
        if record.object_polygons_xy:
            polygons = []
            for object_index, polygon_xy in record.object_polygons_xy.items():
                if object_indices is not None and object_index not in object_indices:
                    continue
                # Label with the canonical concept where the indexer assigned
                # one (M³FD): that is the vocabulary the question and the gold
                # answer are written in, so "person" is what a reviewer needs
                # to see on the box, not the source's raw "People".
                name = (record.object_concepts_by_index.get(object_index)
                        or record.object_names_by_index.get(object_index, ""))
                polygons.append(ScenePolygon(
                    object_index=object_index, name=name,
                    x=[point[0] for point in polygon_xy],
                    y=[point[1] for point in polygon_xy],
                ))
            return polygons
        if not record.annotation_path.is_file():
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
            for index, raw_name in record.object_names_by_index.items()
            if raw_name.lower().replace("_", " ").strip() in normalized_targets
        }

    def object_indices_matching_concepts(self, image_id: str, concepts: set[str]) -> set[int]:
        """Object indices whose canonical `concept` is one of `concepts`.

        M³FD evidence names the class it reasoned about, not the box's index
        (``m3fd_candidates.py``), and the canonical class is not always the
        raw annotation name — the source writes "People" where the vocabulary
        says `person`. So this matches the indexer's assigned concept rather
        than the raw name `object_indices_matching_names` compares.
        """
        record = self.get(image_id)
        if record is None:
            return set()
        normalized_targets = {concept.lower().replace("_", " ").strip() for concept in concepts}
        return {
            index
            for index, concept in record.object_concepts_by_index.items()
            if concept.lower().replace("_", " ").strip() in normalized_targets
        }

    def object_indices_matching_boxes(
        self, image_id: str, boxes: list[list[float]], tolerance: float = 0.5
    ) -> set[int]:
        """Object indices whose thermal box matches one of `boxes`.

        The boxes an M³FD evidence entry carries were copied straight out of
        this same index, so they should agree to the bit; `tolerance` (half a
        pixel) only absorbs the float round-trip through the release CSV's
        JSON. A box that matches nothing is skipped rather than guessed at —
        an overlay must never invent evidence the generator did not record.
        """
        record = self.get(image_id)
        if record is None:
            return set()
        matched: set[int] = set()
        for box in boxes:
            if len(box) != 4:
                continue
            for index, indexed_box in record.object_boxes_by_index.items():
                if all(abs(a - b) <= tolerance for a, b in zip(box, indexed_box)):
                    matched.add(index)
                    break
        return matched
