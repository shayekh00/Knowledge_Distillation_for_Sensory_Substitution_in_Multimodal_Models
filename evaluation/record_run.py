"""Record a run as a self-describing artifact, and index every run (plan §10, §15).

The point of this module is that in three months nobody should have to guess what
`results_val_7b.csv` was. That is not hypothetical — it is exactly what happened
to the previous submission, where two byte-identical prediction files sat under
different names and the modality of a headline figure had to be inferred from a
filename (`docs/New_Submission/legacy_discrepancy_report.md`).

So every run gets a directory that answers, on its own:

    runs/pilot/<run_id>/
      manifest.json        what the run was: model, recipe, seed, modality, mode
      configuration.json   fully resolved settings, hashed into the run id
      predictions.csv      question_id,prediction  (re-scorable against any split)
      metrics.json         machine-readable scores, straight from evaluate.py
      resource_usage.json  peak VRAM, throughput, GPU
      README.md            the same thing in prose, for a human opening the folder

and `runs/INDEX.md` is regenerated from those directories, never hand-edited. §10
requires every table and figure to be generated from saved artifacts rather than
copied by hand; this is what makes that possible.

Usage::

    python evaluation/record_run.py --predictions runs/pilot/B1_depth_val.csv \\
        --recipe B1 --split val --modality depth --model Qwen/Qwen3.5-0.8B \\
        --description "Zero-shot depth reference"
    python evaluation/record_run.py --reindex
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.cache import make_run_id  # noqa: E402

RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
PILOT_ROOT = os.path.join(RUNS_ROOT, "pilot")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

MODEL_SLUGS = {
    "Qwen/Qwen3.5-0.8B": "qw08b",
    "Qwen/Qwen3.5-9B": "qw9b",
    "OpenGVLab/InternVL3_5-1B": "iv1b",
    "OpenGVLab/InternVL3_5-8B": "iv8b",
    "google/gemma-4-12B-it": "gm12b",
}


def slug_for(model: str, teacher: str | None) -> str:
    student = MODEL_SLUGS.get(model, "model")
    return f"{MODEL_SLUGS.get(teacher, 'x')}2{student}" if teacher else f"none2{student}"


def score_predictions(predictions_csv: str, split: str, model_name: str,
                      metrics_path: str) -> dict:
    """Score with the repaired evaluator, in the project venv.

    Deliberately shells out to `evaluate.py` rather than importing it: the
    numbers in a run directory are then produced by exactly the command a reader
    would run themselves, and the evaluator keeps its own environment (it needs
    `inflect`, which the model environment does not have).
    """
    command = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "evaluate.py"),
               "--predictions", predictions_csv, "--split", split,
               "--model-name", model_name, "--json", metrics_path]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"evaluate.py failed:\n{completed.stdout}\n{completed.stderr}")
    with open(metrics_path, encoding="utf-8") as handle:
        return json.load(handle)


def headline(metrics: dict) -> tuple:
    macro = metrics.get("macro_accuracy")
    per_type = metrics.get("per_type") or []
    invalid = sum(entry.get("n_invalid", 0) for entry in per_type)
    items = sum(entry.get("n_items", 0) for entry in per_type)
    return macro, (invalid / items if items else None)


def write_readme(directory: str, meta: dict, metrics: dict) -> None:
    macro, invalid_rate = headline(metrics)
    lines = [f"# {meta['run_id']}", "",
             f"**{meta['description']}**", "",
             "| Field | Value |", "|---|---|",
             f"| Recipe | `{meta['recipe']}` |",
             f"| Status | **{'PILOT' if meta['pilot'] else 'CONFIRMATORY'}** |",
             f"| Split | `{meta['split']}` |",
             f"| Student model | `{meta['model']}` |",
             f"| Teacher model | `{meta.get('teacher') or '—'}` |",
             f"| Inference modality | `{meta['modality']}` |",
             f"| Distillation mode | `{meta['distillation_mode']}` |",
             f"| Prompt style | `{meta.get('prompt_style') or '—'}` |",
             f"| Seed | {meta['seed']} |",
             f"| Recorded (UTC) | {meta['recorded_utc']} |", ""]
    if macro is not None:
        lines += [f"**Macro accuracy: {macro:.1%}**"
                  + (f"  ·  invalid outputs: {invalid_rate:.1%}"
                     if invalid_rate is not None else ""), ""]
        lines += ["| Type | n | accuracy | invalid |", "|---|---:|---:|---:|"]
        for entry in metrics.get("per_type") or []:
            n_items = entry["n_items"]
            lines.append(
                f"| {entry['question_type']} | {n_items} | {entry['accuracy']:.1%} "
                f"| {entry.get('n_invalid', 0) / n_items:.1%} |")
        lines.append("")
    baselines = metrics.get("baselines") or {}
    if baselines:
        lines += ["Reference baselines on this split (macro): "
                  + ", ".join(
                      f"{name} {sum(values.values()) / len(values):.1%}"
                      for name, values in baselines.items() if values), ""]
    if meta.get("pilot"):
        lines += ["> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of "
                  "`docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory "
                  "result and must not appear in a main or ablation table.", ""]
    if meta.get("notes"):
        lines += ["## Notes", "", meta["notes"], ""]
    lines += ["## Reproduce", "", "```bash",
              f"python evaluate.py --predictions runs/pilot/{meta['run_id']}/predictions.csv \\",
              f"    --split {meta['split']} --model-name \"{meta['description']}\"",
              "```", ""]
    with open(os.path.join(directory, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def record(args) -> str:
    configuration = {
        "recipe": args.recipe, "model": args.model, "teacher": args.teacher,
        "modality": args.modality, "split": args.split, "seed": args.seed,
        "prompt_style": args.prompt_style, "representation": args.representation,
        "distillation_mode": args.distillation_mode, "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank, "epochs": args.epochs,
    }
    run_id = make_run_id(datetime.now(timezone.utc).strftime("%Y%m%d"),
                         slug_for(args.model, args.teacher), args.recipe,
                         args.seed, configuration)
    directory = os.path.join(PILOT_ROOT if args.pilot else RUNS_ROOT, run_id)
    os.makedirs(directory, exist_ok=True)

    shutil.copy(args.predictions, os.path.join(directory, "predictions.csv"))
    metrics = score_predictions(os.path.join(directory, "predictions.csv"), args.split,
                                args.description,
                                os.path.join(directory, "metrics.json"))

    meta = dict(configuration)
    meta.update({"run_id": run_id, "description": args.description, "pilot": args.pilot,
                 "notes": args.notes,
                 "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    with open(os.path.join(directory, "configuration.json"), "w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2, sort_keys=True)
    if args.resources and os.path.isfile(args.resources):
        shutil.copy(args.resources, os.path.join(directory, "resource_usage.json"))

    write_readme(directory, meta, metrics)
    return run_id


def reindex() -> str:
    """Regenerate `runs/INDEX.md` from the run directories. Never hand-edited."""
    entries = []
    for root in (PILOT_ROOT, RUNS_ROOT):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            manifest_path = os.path.join(root, name, "manifest.json")
            metrics_path = os.path.join(root, name, "metrics.json")
            if not os.path.isfile(manifest_path):
                continue
            with open(manifest_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            metrics = {}
            if os.path.isfile(metrics_path):
                with open(metrics_path, encoding="utf-8") as handle:
                    metrics = json.load(handle)
            macro, invalid_rate = headline(metrics)
            # Relative to RUNS_ROOT, because INDEX.md lives there — relative to the
            # project root the links would resolve one level too high.
            entries.append((meta, macro, invalid_rate,
                            os.path.relpath(os.path.join(root, name), RUNS_ROOT)))

    lines = ["# Run index", "",
             "Generated by `python evaluation/record_run.py --reindex`. **Do not edit "
             "by hand** — it is regenerated from each run's `manifest.json` and "
             "`metrics.json`, so it can never drift from the artifacts it describes.", "",
             "Every row links to a self-describing directory: what the run was, its "
             "predictions, its scores, and its resource usage.", "",
             "| Run | Recipe | Status | Split | Modality | Mode | Macro | Invalid | Description |",
             "|---|---|---|---|---|---|---:|---:|---|"]
    for meta, macro, invalid_rate, path in sorted(
            entries, key=lambda item: (item[0].get("recipe", ""), item[0].get("run_id", ""))):
        cells = [
            f"[`{meta['run_id']}`]({path}/)",
            f"`{meta.get('recipe', '?')}`",
            "PILOT" if meta.get("pilot") else "CONFIRMATORY",
            str(meta.get("split", "?")),
            str(meta.get("modality", "?")),
            str(meta.get("distillation_mode", "none")),
            f"{macro:.1%}" if macro is not None else "—",
            f"{invalid_rate:.1%}" if invalid_rate is not None else "—",
            str(meta.get("description", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Reading these numbers", "",
              "* **PILOT** rows were produced on a 16 GB RTX 4080 SUPER under Option A "
              "of `docs/New_Submission/experiment_protocol.md` §9.5. They establish "
              "feasibility and catch defects. They are **not** confirmatory results and "
              "must not appear in a main or ablation table.",
              "* `Invalid` is the share of predictions that are not a legal answer for "
              "their row. A high value means the run is measuring output formatting "
              "rather than perception — read it before reading the accuracy.",
              "* All accuracies come from the repaired `evaluate.py`, which counts "
              "unanswered items as wrong and rejects duplicate question ids.",
              "* `val` is the development split. `test` is evaluated once, after "
              "settings are locked (protocol §5).", ""]
    os.makedirs(RUNS_ROOT, exist_ok=True)
    path = os.path.join(RUNS_ROOT, "INDEX.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reindex", action="store_true", help="Regenerate runs/INDEX.md.")
    parser.add_argument("--predictions")
    parser.add_argument("--recipe")
    parser.add_argument("--description", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher")
    parser.add_argument("--modality", default="depth")
    parser.add_argument("--representation", default="replicated")
    parser.add_argument("--distillation-mode", default="none")
    parser.add_argument("--prompt-style", default="terse")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resources", help="resource_usage.json to copy in.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--confirmatory", dest="pilot", action="store_false")
    parser.set_defaults(pilot=True)
    args = parser.parse_args()

    if args.reindex:
        print(f"wrote {reindex()}")
        return
    if not (args.predictions and args.recipe):
        raise SystemExit("--predictions and --recipe are required (or pass --reindex)")
    run_id = record(args)
    print(f"recorded {run_id}")
    print(f"wrote {reindex()}")


if __name__ == "__main__":
    main()
