import pytest

from tools.audit_app.sources import SOURCES, active_source, resolve_source


def test_defaults_to_sunrgbd_and_keeps_its_original_paths():
    source = resolve_source(None)

    assert source.name == "sunrgbd"
    assert source.scene_index == "data/index/scene_index.jsonl"
    assert source.canonical_objects == "data/vocab/canonical_objects.csv"
    assert source.audit_dir == "audit"
    # The pre-`--source` behaviour: one image, sensor strata, SUN-RGB-D
    # evidence, no display rotation.
    assert source.modalities == ("rgb",)
    assert source.stratify_by == "sensor"
    assert source.evidence_style == "sunrgbd"
    assert source.display_rotation is False


def test_m3fd_is_thermal_first_and_unstratified():
    source = resolve_source("m3fd")

    # Thermal is what the student sees, so it is what gold must be true of.
    assert source.modalities == ("thermal", "rgb")
    assert source.default_modality == "thermal"
    # No `sensor` column exists in build_release_m3fd.COLUMNS to stratify on.
    assert source.stratify_by is None
    assert source.evidence_style == "m3fd"
    assert source.image_column == "thermal_path"
    assert source.display_rotation is False


def test_only_arkitscenes_asks_for_display_rotation():
    rotating = {name for name, source in SOURCES.items() if source.display_rotation}

    assert rotating == {"arkitscenes"}


def test_unknown_source_names_the_valid_ones():
    with pytest.raises(SystemExit) as raised:
        resolve_source("m3df")  # transposed, the plausible typo

    assert "m3fd" in str(raised.value) and "sunrgbd" in str(raised.value)


def test_source_name_is_case_and_space_insensitive():
    assert resolve_source("  M3FD ").name == "m3fd"


def test_active_source_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("AUDIT_SOURCE", "m3fd")

    assert active_source().name == "m3fd"


def test_active_source_defaults_when_env_var_absent(monkeypatch):
    monkeypatch.delenv("AUDIT_SOURCE", raising=False)

    assert active_source().name == "sunrgbd"


def test_paths_resolve_under_the_project_root():
    source = resolve_source("m3fd")

    assert source.scene_index_path.name == "scene_index_m3fd.jsonl"
    assert source.scene_index_path.is_absolute()
    assert source.audit_dir_path.name == "audit_m3fd"
    assert source.release_test_csv_path.parts[-3:] == (
        "VQA-M3FD-Thermal-v1", "rule_based", "test.csv")


def test_modality_validation_is_per_source():
    assert resolve_source("m3fd").is_known_modality("thermal")
    # SUN-RGB-D has no thermal frame; asking for one is an error, not a
    # silent fallback to RGB.
    assert not resolve_source("sunrgbd").is_known_modality("thermal")
