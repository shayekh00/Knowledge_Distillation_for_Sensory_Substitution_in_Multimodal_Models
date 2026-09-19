"""Named audit sources: which release, index, vocabulary and imagery to serve.

Until now a non-SUN-RGB-D dataset was audited by setting `SCENE_INDEX_JSONL`,
`CANONICAL_OBJECTS_CSV` and `AUDIT_DIR` by hand (main.py's original note,
arkitscenes_plan.md §6 Phase 4). That is enough for ARKitScenes, whose rows are
SUN-RGB-D-shaped, but not for M³FD, which differs in ways no path override can
express:

* two registered images per record — `rgb_path` **and** `thermal_path` — where
  every other source has one, and the reviewer must be able to flip between
  them (M3FD_THERMAL_VQA_RUNBOOK.md: audit "against both source modalities and
  the annotation overlay");
* geometry stored in the *thermal* frame, so an overlay drawn over the RGB
  image has to be rescaled into the RGB frame first;
* evidence that names concepts and boxes rather than object indices
  (`m3fd_candidates.py`), so the SUN-RGB-D index-extraction path finds nothing;
* release rows with no `sensor` column, so the audit sampler cannot stratify on
  one.

So the dataset choice becomes a single named `source` that carries those
differences as data. Selection order everywhere is: explicit `--source` /
`AUDIT_SOURCE`, then the individual env vars, which still override whatever the
source declares so existing invocations keep working unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_NAME = "sunrgbd"
SOURCE_ENV_VAR = "AUDIT_SOURCE"


@dataclass(frozen=True)
class AuditSource:
    """Everything about a dataset that the reviewer tooling has to vary.

    Paths are repo-relative strings here and resolved against `PROJECT_ROOT` by
    the properties below, so this table stays readable and a test can point the
    whole thing at a tmpdir by overriding the env vars instead.
    """

    name: str
    title: str
    scene_index: str
    canonical_objects: str
    release_test_csv: str
    audit_dir: str
    # Display modalities, most useful first; the first is what loads by default.
    # A single-entry tuple means the UI shows no modality switch at all.
    modalities: tuple[str, ...] = ("rgb",)
    # Release column the audit sample is stratified on within each question
    # type. None means "no stratification" — draw the per-type quota uniformly.
    stratify_by: str | None = "sensor"
    # Which evidence dialect the release's `evidence` JSON speaks. See
    # audit_items.resolve_evidence_object_indices.
    evidence_style: str = "sunrgbd"
    # Whether frames need ARKitScenes' per-frame upright correction on display.
    display_rotation: bool = False
    # Release column holding the image the optional model triage pass should
    # be shown (model_pass.py). For M³FD this is the thermal frame, never the
    # RGB one: the gold label has to be true of what the student sees.
    image_column: str = "image_path"

    @property
    def scene_index_path(self) -> Path:
        return PROJECT_ROOT / self.scene_index

    @property
    def canonical_objects_path(self) -> Path:
        return PROJECT_ROOT / self.canonical_objects

    @property
    def release_test_csv_path(self) -> Path:
        return PROJECT_ROOT / self.release_test_csv

    @property
    def audit_dir_path(self) -> Path:
        return PROJECT_ROOT / self.audit_dir

    @property
    def default_modality(self) -> str:
        return self.modalities[0]

    def is_known_modality(self, modality: str) -> bool:
        return modality in self.modalities


SOURCES: dict[str, AuditSource] = {
    "sunrgbd": AuditSource(
        name="sunrgbd",
        title="VQA-SUNRGBD-v2",
        scene_index="data/index/scene_index.jsonl",
        canonical_objects="data/vocab/canonical_objects.csv",
        release_test_csv="release/VQA-SUNRGBD-v2/rule_based/test.csv",
        audit_dir="audit",
    ),
    "arkitscenes": AuditSource(
        name="arkitscenes",
        title="VQA-ARKitScenes-v1",
        scene_index="data/index/scene_index_arkit.jsonl",
        canonical_objects="data/vocab_arkit/canonical_objects.csv",
        release_test_csv="release/VQA-ARKitScenes-v1/rule_based/test.csv",
        audit_dir="audit_arkit",
        display_rotation=True,
    ),
    "m3fd": AuditSource(
        name="m3fd",
        title="VQA-M3FD-Thermal-v1",
        scene_index="data/index/scene_index_m3fd.jsonl",
        canonical_objects="data/vocab_m3fd/canonical_objects.csv",
        release_test_csv="release/VQA-M3FD-Thermal-v1/rule_based/test.csv",
        audit_dir="audit_m3fd",
        # Thermal first: it is the only image the student ever sees, so it is
        # the one the gold label has to be true of. RGB is the registration
        # cross-check the runbook asks the reviewer to make, not the subject.
        modalities=("thermal", "rgb"),
        # Release rows carry `sequence_id`, never `sensor` (build_release_m3fd.
        # COLUMNS), and every M³FD frame is the same thermal sensor anyway.
        stratify_by=None,
        evidence_style="m3fd",
        image_column="thermal_path",
    ),
}


def resolve_source(name: str | None) -> AuditSource:
    """Look up a source by name, defaulting to SUN-RGB-D."""
    resolved_name = (name or DEFAULT_SOURCE_NAME).strip().lower()
    try:
        return SOURCES[resolved_name]
    except KeyError:
        known = ", ".join(sorted(SOURCES))
        raise SystemExit(f"unknown audit source {resolved_name!r}; known sources: {known}") from None


def active_source() -> AuditSource:
    """The source the server process was started for (`AUDIT_SOURCE=m3fd`)."""
    return resolve_source(os.environ.get(SOURCE_ENV_VAR))
