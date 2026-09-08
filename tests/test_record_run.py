"""The run id must distinguish experiments that differ (plan §10, §15).

Two runs whose settings differ must not share an id, because `record_run` writes
into a directory named by it and a collision overwrites the evidence behind a
number. That is not hypothetical: on 2026-09-06 it happened twice in one day,
once for `batch_size` (a batch-4 run overwrote the batch-1 run it was meant to be
compared against) and once for `precision` (bf16 teacher rows hashed identically
to the NF4 ones). Both times the cause was the same — a field that defines an
experiment was missing from a hand-maintained list.

So these tests pin the *general* property rather than the two fields that
bit us, and the first one fails whenever a switch is added to `RecipeConfig`
without reaching the id.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.cache import make_run_id  # noqa: E402
from distillation.runner import RecipeConfig  # noqa: E402
from evaluation.record_run import build_configuration, check_pilot_agreement  # noqa: E402


def make_args(**overrides):
    defaults = dict(
        recipe="X2", model="Qwen/Qwen3.5-0.8B", teacher="Qwen/Qwen3.5-9B",
        modality="depth", split="val", seed=17, prompt_style="terse",
        representation="replicated", distillation_mode="xtoken",
        learning_rate=2e-5, lora_rank=16, epochs=1, precision="bfloat16",
        resources=None, recipe_config=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def run_id_for(configuration):
    return make_run_id("20260906", "qw9b2qw08b", configuration["recipe"],
                       configuration["seed"], configuration)


def test_run_id_covers_every_recipe_config_field(tmp_path):
    """Every switch that defines a training row must reach the hashed config.

    This is the guard the two 2026-09-06 collisions needed. `RecipeConfig` is a
    dataclass and `resolved()` is `asdict()`, so a new switch is picked up
    automatically — unless someone reintroduces a hand-maintained list, which is
    what this test exists to catch.
    """
    recipe = RecipeConfig(recipe="X2", kd_objective="xtoken", top_k=4096,
                          kd_temperature=2.0, use_ce=True)
    path = write_json(tmp_path, "recipe.json", recipe.resolved())

    configuration = build_configuration(make_args(recipe_config=path))

    missing = set(recipe.resolved()) - set(configuration)
    assert not missing, (
        f"{sorted(missing)} define a training row but never reach the run id, so two "
        f"runs differing only in them would collide and overwrite each other")


@pytest.mark.parametrize("field, value", [
    ("kd_temperature", 4.0),
    ("top_k", 8192),
    ("lambda_kd", 0.5),
    ("kd_objective", "candidate"),
    ("feature_objective", "cosine"),
    ("use_loca", True),
    ("stage", "F"),
    ("trainable_modules", ["language_attention", "projector"]),
])
def test_changing_any_kd_switch_changes_the_run_id(tmp_path, field, value):
    """Perturbing one switch must move the id. Covers the KD knobs specifically."""
    base = RecipeConfig(recipe="X2", kd_objective="xtoken", top_k=4096).resolved()
    altered = dict(base, **{field: value})

    base_id = run_id_for(build_configuration(
        make_args(recipe_config=write_json(tmp_path, "a.json", base))))
    altered_id = run_id_for(build_configuration(
        make_args(recipe_config=write_json(tmp_path, "b.json", altered))))

    assert base_id != altered_id, f"{field} does not affect the run id"


@pytest.mark.parametrize("field, value", [
    ("precision", "nf4"),
    ("representation", "gradient"),
    ("learning_rate", 1e-5),
    ("modality", "rgb"),
    ("prompt_style", "enumerated"),
])
def test_changing_any_inference_field_changes_the_run_id(field, value):
    """The same property for the fields a zero-shot row is defined by."""
    base_id = run_id_for(build_configuration(make_args()))
    altered_id = run_id_for(build_configuration(make_args(**{field: value})))
    assert base_id != altered_id, f"{field} does not affect the run id"


@pytest.mark.parametrize("field, value", [
    ("batch_size", 1),
    ("effective_batch", 32),
    ("accumulation", 16),
    ("gradient_checkpointing", True),
    ("max_epochs", 5),
    ("patience", 1),
])
def test_changing_the_batch_schedule_changes_the_run_id(tmp_path, field, value):
    """The `batch_size` collision itself, pinned as a regression test.

    `max_epochs`/`patience` are the same class of gap, caught before it could
    repeat: two runs under a different §7.3 stopping policy — say patience 1
    vs 2 — would otherwise hash identically and silently overwrite each
    other, exactly like the original `batch_size` collision.
    """
    base = {"batch_size": 4, "effective_batch": 16, "accumulation": 4,
            "gradient_checkpointing": False, "max_epochs": 10, "patience": 2}
    altered = dict(base, **{field: value})

    base_id = run_id_for(build_configuration(make_args(
        resources=write_json(tmp_path, "ra.json", {"resources": base}))))
    altered_id = run_id_for(build_configuration(make_args(
        resources=write_json(tmp_path, "rb.json", {"resources": altered}))))

    assert base_id != altered_id, f"{field} does not affect the run id"


def test_recipe_config_contradicting_the_command_line_is_refused(tmp_path):
    """Silent drift between two sources of truth is worse than a crash."""
    recipe = RecipeConfig(recipe="X2", kd_objective="xtoken", top_k=4096,
                          seed=999).resolved()
    path = write_json(tmp_path, "recipe.json", recipe)

    with pytest.raises(SystemExit, match="disagrees with the command line"):
        build_configuration(make_args(recipe_config=path, seed=17))


def test_confirmatory_training_recorded_as_pilot_is_refused(tmp_path):
    """A run trained with --confirmatory must not silently land in runs/pilot/.

    §9.5 is binding: "no PILOT number enters a main or ablation table." `pilot`
    is deliberately not part of the run-id hash (it is a label, not an
    experiment-defining field), so a mismatch here would not collide on its
    own — it would just record wrong. This is what would have let today's
    corrected B3/B5 confirmatory reruns land in runs/pilot/ if record_run.py
    were called without --confirmatory to match.
    """
    resources = write_json(tmp_path, "resource_usage.json",
                           {"resources": {"pilot": False}})
    with pytest.raises(SystemExit, match="was trained with pilot=False"):
        check_pilot_agreement(resources, True)

    # Agreement in either direction is silent.
    check_pilot_agreement(resources, False)
    pilot_resources = write_json(tmp_path, "pilot_resource_usage.json",
                                 {"resources": {"pilot": True}})
    check_pilot_agreement(pilot_resources, True)
    with pytest.raises(SystemExit, match="was trained with pilot=True"):
        check_pilot_agreement(pilot_resources, False)

    # No --resources at all (zero-shot rows) has nothing to check against.
    check_pilot_agreement(None, True)
