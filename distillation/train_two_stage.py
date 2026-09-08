"""Chain a two-stage row: stage F (feature alignment) then stage S2 (answers).

§13.1 point 5 was resolved on 2026-09-07 in favour of two stages, on the reading
the protocol already contained: §7.1 describes D3–D9 as two-stage rows where
"stage one is a separate training pass before S2 begins", and §8.2 defines S2 as
vision-frozen. A row's feature objective therefore belongs to its F stage. This
is also the author's own two-phase method.

Run as **two processes, not two function calls**, on purpose. The stage-F model,
its optimizer state, and its alignment head are all resident when stage two
needs to load a fresh base model and merge stage one's adapter into it; doing
that in one process means holding both at once for no reason. A stage boundary is
also the natural place for the run to be resumable — if S2 fails, stage F's
checkpoint is already on disk and `--skip-f` picks up from it rather than
repeating a training pass.

The two stages are deliberately *not* given a shared epoch budget. Stage one (F
or P) runs a declared fixed number of epochs (`--alignment-epochs`) and keeps
its last, rather than early-stopping on its own full-val reading — that reading
exists (a pure-F row does score the whole val split every epoch; corrected
2026-09-07 after an earlier, wrong claim that it could not — see
experiment_protocol.md §13) but is diagnostic, not selective: it measures how
legible the shifted vision is to a language model that has not yet adapted to
it, which can diverge from how good a starting point the shift is once stage
two adapts. Stage two keeps §7.3's 10-epoch/patience-2 rule scored on
validation macro. Model selection happens once, in the stage whose reading is
trusted for it.

Usage::

    python distillation/train_two_stage.py --recipe D8 \\
        --feature-cache <pooled_features dir> --cache <topk_logits dir> \\
        --out runs/kd/D8_depth_s17
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.runner import recipe_library  # noqa: E402

TRAIN_KD = os.path.join(PROJECT_ROOT, "distillation", "train_kd.py")


def run_stage(label: str, arguments: list) -> None:
    printable = " ".join(arguments)
    print(f"\n=== {label} ===\n{printable}\n", flush=True)
    completed = subprocess.run([sys.executable, TRAIN_KD] + arguments, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}; "
                         f"stage two is not started on a failed stage one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True,
                        help="A two-stage row: one declaring a feature objective "
                             "(D3, D4, D5, D6, D8, D9 — not D7, which is joint).")
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--cache", help="Top-K logits cache. Required for rows whose "
                        "S2 has a KD term, and for D6, whose stage P also has one.")
    parser.add_argument("--out", required=True,
                        help="Base directory; stages land in <out>/stage_F and <out>/stage_S2.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--modality", default="depth")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alignment-epochs", "--f-epochs", type=int, default=3,
                        dest="alignment_epochs",
                        help="How many passes the vision-alignment stage gets. "
                             "(--f-epochs is accepted as an alias; 'F' is the "
                             "protocol's stage id, not a description.)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Answer stage (S2) max epochs.")
    parser.add_argument("--patience", type=int, default=2, help="Stage-two patience.")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--alignment-learning-rate", "--f-learning-rate", type=float,
                        dest="alignment_learning_rate",
                        help="Vision-alignment stage learning rate; defaults to "
                             "--learning-rate.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--negative-bank", type=int, default=255)
    parser.add_argument("--limit", type=int, help="First N train rows (smoke test).")
    parser.add_argument("--val-limit", type=int, help="Stage-two val rows (smoke test).")
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--skip-alignment", "--skip-f", action="store_true",
                        dest="skip_alignment",
                        help="Reuse an existing <out>/stage_F (or stage_P) rather than "
                             "retraining it.")
    parser.add_argument("--p-lambda-kd", type=float, default=0.1,
                        help="D6 only: weight of stage P's raw KD term (§8.2).")
    parser.add_argument("--confirmatory", dest="pilot", action="store_false")
    parser.set_defaults(pilot=True)
    args = parser.parse_args()

    library = recipe_library()
    if args.recipe not in library:
        parser.error(f"unknown recipe {args.recipe!r}; have {sorted(library)}")
    config = library[args.recipe]
    if not config.is_two_stage():
        parser.error(
            f"recipe {args.recipe} declares no feature objective, so it has no stage "
            f"one. Run it directly with train_kd.py.")

    stage_two_config = config.stage_two("placeholder")
    needs_logits = stage_two_config.kd_objective not in ("none", "sequence")
    if needs_logits and not args.cache:
        parser.error(f"{args.recipe}'s stage two has a "
                     f"{stage_two_config.kd_objective} KD term and needs --cache")

    # D6 is the one row whose stage one is P (feature alignment + a small raw
    # KD term, §8.2) rather than F (feature alignment only). Every other
    # two-stage row uses F. Directory name follows the stage actually run.
    stage_one_flag = "P" if args.recipe == "D6" else "F"
    f_directory = os.path.join(args.out, f"stage_{stage_one_flag}")
    s2_directory = os.path.join(args.out, "stage_S2")
    common = ["--recipe", args.recipe, "--model", args.model, "--teacher", args.teacher,
              "--modality", args.modality, "--seed", str(args.seed),
              "--batch-size", str(args.batch_size),
              "--effective-batch", str(args.effective_batch),
              "--lora-rank", str(args.lora_rank), "--log-every", str(args.log_every)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if not args.pilot:
        common += ["--confirmatory"]

    if args.skip_alignment:
        if not os.path.isdir(os.path.join(f_directory, "adapter")):
            parser.error(f"--skip-alignment but {f_directory}/adapter does not exist")
        print(f"reusing the vision-alignment stage at {f_directory}", flush=True)
    else:
        stage_one_args = common + [
            "--stage", stage_one_flag, "--feature-cache", args.feature_cache,
            "--negative-bank", str(args.negative_bank),
            "--epochs", str(args.alignment_epochs),
            "--learning-rate", str(args.alignment_learning_rate or args.learning_rate),
            "--out", f_directory]
        if stage_one_flag == "P":
            # P keeps kd_objective set (§8.2), so — unlike a pure F row — it
            # needs the same top-K logit cache stage two does.
            if not args.cache:
                parser.error("D6's stage one is P, which has a raw KD term and "
                             "needs --cache (the top-K logits cache)")
            stage_one_args += ["--cache", args.cache, "--p-lambda-kd", str(args.p_lambda_kd)]
        run_stage(f"{args.recipe} vision alignment (stage {stage_one_flag}) — "
                  f"{config.feature_objective}"
                  + (" + raw KD" if stage_one_flag == "P" else ""),
                  stage_one_args)

    stage_two = common + ["--stage", "S2", "--parent-checkpoint", f_directory,
                          "--feature-cache", args.feature_cache,
                          "--epochs", str(args.epochs), "--patience", str(args.patience),
                          "--learning-rate", str(args.learning_rate),
                          "--out", s2_directory]
    if args.cache:
        stage_two += ["--cache", args.cache]
    if args.val_limit:
        stage_two += ["--val-limit", str(args.val_limit)]
    run_stage(f"{args.recipe} answer stage (S2) — vision frozen", stage_two)

    summary = {"recipe": args.recipe, "seed": args.seed,
               "stage_f": f_directory, "stage_s2": s2_directory,
               "alignment_epochs": args.alignment_epochs, "s2_max_epochs": args.epochs,
               "s2_patience": args.patience,
               "feature_objective": config.feature_objective,
               "adapter_to_evaluate": os.path.join(s2_directory, "adapter")}
    for stage, directory in (("stage_f", f_directory), ("stage_s2", s2_directory)):
        path = os.path.join(directory, "resource_usage.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                summary[f"{stage}_resources"] = json.load(handle)["resources"]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "two_stage_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n=== {args.recipe} complete ===", flush=True)
    print(f"evaluate: {summary['adapter_to_evaluate']}", flush=True)


if __name__ == "__main__":
    main()
