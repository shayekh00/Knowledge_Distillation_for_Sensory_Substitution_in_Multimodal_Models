"""Verify X-Token's projection is correct on real Qwen3.5-9B/0.8B data.

Not a unit test on synthetic fixtures (those already exist in
`tests/test_teacher_cache_loader.py` and `tests/test_xtoken.py`) — this checks
the actual deployed code against actual cached teacher signals and an actual
student forward pass, because the earlier investigation found something a
synthetic fixture cannot: the real Qwen9B/Qwen0.8B vocabulary mapping is NOT a
literal identity even though both are "the same tokenizer family" — 944 of
248,077 ids (0.38%) differ, all byte-fallback tokens that collapse many
student ids onto one teacher id. That collapse only exists in the real
tokenizer files, so only a check against real data can catch a bug in how it's
handled.

Two independent things are checked, both against `distillation.xtoken`'s
actual `project_student_probs` / `projected_kl_loss` — "independent" meaning a
different computation path, not a reformatted copy of the same code:

1. **The scatter-add accumulates rather than overwrites.** `project_student_probs`
   uses `scatter_add_`, the correct primitive for "sum every student token's
   mass that maps to this teacher token" — but the right primitive doesn't
   rule out an indexing or dimension bug. Verified here with a Python dict
   groupby, not another tensor op.
2. **The end-to-end KD loss matches an independent re-derivation** on real
   student logits from a real forward pass over real cached examples, using
   direct indexing instead of the general sparse-projection machinery.

If either check fails, every X2 (and any other xtoken-objective) result on
this pair is suspect and must not be reported — this is exactly the class of
defect `NEW_SUBMISSION.md` §4.1 records happening silently before.
"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.cache import CacheKey  # noqa: E402
from distillation.losses import IGNORE_INDEX, shift_for_causal_lm, valid_answer_mask  # noqa: E402
from distillation.teacher_cache_loader import TeacherCache  # noqa: E402
from distillation.train_student import build_batch, build_image, load_rows  # noqa: E402
from distillation.xtoken import VocabularyMapping, project_student_probs, projected_kl_loss  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "checkpoints_scratch", "teacher_cache",
                         "topk_logits_0462bee1d6f00444")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "checkpoints_scratch", "xtoken_mappings",
                           "xtoken_mapping_86ba8884a7ad4552.pt")
N_EXAMPLES = 6


def independent_projection(student_probs: torch.Tensor, student_ids: torch.Tensor,
                           teacher_ids: torch.Tensor, teacher_vocab_size: int) -> torch.Tensor:
    """The same projection as `project_student_probs`, computed by a Python
    dict groupby instead of `scatter_add_` — a genuinely different mechanism,
    so agreement is real evidence rather than two paths sharing one bug.
    """
    groups: dict[int, list[int]] = {}
    for student_id, teacher_id in zip(student_ids.tolist(), teacher_ids.tolist()):
        groups.setdefault(teacher_id, []).append(student_id)

    projected = student_probs.new_zeros((student_probs.size(0), teacher_vocab_size))
    for teacher_id, student_id_group in groups.items():
        projected[:, teacher_id] = student_probs[:, student_id_group].sum(dim=1)
    return projected


def check_scatter_add_accumulates(mapping) -> None:
    """The specific case the real mapping actually has: multiple student ids
    (byte-fallback tokens 95..102) collapsing onto one teacher id (94)."""
    student_ids, teacher_ids = mapping.student_ids, mapping.teacher_ids
    teacher_id_counts: dict[int, int] = {}
    for teacher_id in teacher_ids.tolist():
        teacher_id_counts[teacher_id] = teacher_id_counts.get(teacher_id, 0) + 1
    collapsed_teacher_id = max(teacher_id_counts, key=teacher_id_counts.get)
    n_collapsed = teacher_id_counts[collapsed_teacher_id]
    print(f"most-collapsed teacher id: {collapsed_teacher_id} "
          f"receives {n_collapsed} distinct student ids")
    if n_collapsed < 2:
        raise SystemExit(
            "expected a real many-to-one collapse in this mapping (byte-fallback "
            "tokens) and found none — the earlier investigation's finding did not "
            "reproduce; do not trust the rest of this check without understanding why")

    contributing_student_ids = [s for s, t in zip(student_ids.tolist(), teacher_ids.tolist())
                                if t == collapsed_teacher_id]
    probs = torch.zeros((1, mapping.student_vocab_size))
    for student_id in contributing_student_ids:
        probs[0, student_id] = 1.0 / len(contributing_student_ids)  # equal mass, sums to 1
    projected = project_student_probs(probs, mapping)
    got = projected[0, collapsed_teacher_id].item()
    print(f"  scatter_add_ result at collapsed id: {got:.6f} (expected 1.000000 "
          f"if all {len(contributing_student_ids)} contributions were summed)")
    # Tolerance scales with how many terms are being summed: this is ordinary
    # float32 accumulation noise, not evidence about scatter_add_ specifically.
    # Measured on Gemma-4-12B-it -> Qwen3.5-0.8B (2026-09-07): a 5,880-way
    # collapse (vs. 665 for the Qwen-Qwen pair this constant was first tuned
    # on) landed at 0.999972, a 2.8e-5 error that a fixed 1e-5 tolerance would
    # wrongly reject. Confirmed as float32 noise, not a bug, three ways: (1) a
    # float64 rerun of the identical scatter_add_ call lands at 0.99999999999993
    # (6.6e-14 error); (2) a plain sequential Python float32 sum of n copies of
    # 1/n reproduces the *exact same* 0.9999720454216003 as scatter_add_,
    # bit-for-bit; (3) the same sum in float64 or with Python's arbitrary-
    # precision float matches to 1e-16. `n * 1e-8` is a generous bound for
    # float32 summation of n terms of similar magnitude (empirically ~5e-9/term
    # here); `max(1e-5, ...)` keeps the original constant as a floor so a
    # small-collapse pair is held to the tighter bar it was verified against.
    tolerance = max(1e-5, n_collapsed * 1e-8)
    if abs(got - 1.0) > tolerance:
        raise SystemExit(
            f"scatter_add_ did not sum all contributions onto the collapsed teacher "
            f"id — got {got}, expected 1.0 within tolerance {tolerance:.2e}. This means "
            f"every real X2 result on this pair understates the projected probability "
            f"at collapsed ids.")
    print(f"  PASS: scatter_add_ correctly sums all contributions, not just the last "
          f"one (tolerance {tolerance:.2e} for {n_collapsed} terms)")


def main() -> None:
    print(f"loading mapping: {MAPPING_PATH}")
    # Saved as a plain dict by load_or_build_mapping (torch.save({...}));
    # reconstructing via the same idiom that function itself uses on load.
    mapping = VocabularyMapping(**torch.load(MAPPING_PATH, weights_only=False))
    check_scatter_add_accumulates(mapping)

    print("\nloading student model (CPU, float32 — GPU is occupied by the running "
          "multi-epoch training) and processor", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
    model.eval()

    key = CacheKey({
        "dataset_version": "v2.4", "split": "train", "teacher_model": "Qwen/Qwen3.5-9B",
        "teacher_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "teacher_tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "precision": "bfloat16", "prompt_hash": "f6cc1b602803c6f1",
        "rgb_transform": "PIL RGB, processor default resize",
        "signal_kind": "topk_logits", "top_k": 4096, "temperature": 1.0,
    })
    cache = TeacherCache(CACHE_DIR, key)
    rows = load_rows("train", limit=N_EXAMPLES)

    print(f"\nrunning a real forward pass over {N_EXAMPLES} real cached examples "
          f"(this takes a minute or two on CPU)", flush=True)
    images = [build_image(row, "rgb", "replicated") for row in rows]
    batch = build_batch(processor, rows, images)
    labels = batch.pop("labels")
    with torch.inference_mode():
        logits = model(**batch).logits

    shifted_logits, shifted_labels = shift_for_causal_lm(logits, labels)
    mask = valid_answer_mask(shifted_labels).reshape(-1)
    answer_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))[mask]
    vocabulary = mapping.student_vocab_size
    student_probs = torch.softmax(answer_logits[:, :vocabulary].float(), dim=-1)
    print(f"real supervised positions in this batch: {student_probs.size(0)}")

    teacher = cache.signals_for([row["question_id"] for row in rows])
    n_teacher = teacher.topk_ids.size(0)
    if n_teacher != student_probs.size(0):
        raise SystemExit(
            f"teacher rows ({n_teacher}) != student rows ({student_probs.size(0)}) — "
            f"cannot run the comparison; something about the batch construction here "
            f"disagrees with train_kd.py's own path")

    production = projected_kl_loss(teacher.topk_ids, teacher.topk_probs, student_probs, mapping)

    # Independent re-derivation: different projection mechanism (dict groupby,
    # not scatter_add_), same formula from there.
    independent_projected = independent_projection(
        student_probs.double(), mapping.student_ids, mapping.teacher_ids, mapping.teacher_vocab_size)
    independent_student_at_k = independent_projected.gather(1, teacher.topk_ids)
    independent_student_at_k = (independent_student_at_k
                                / independent_student_at_k.sum(-1, keepdim=True).clamp_min(1e-12))
    teacher_probs = teacher.topk_probs.double()
    teacher_probs = teacher_probs / teacher_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    independent_kl = (teacher_probs * (teacher_probs.clamp_min(1e-12).log()
                                       - independent_student_at_k.clamp_min(1e-12).log())
                      ).sum(-1).mean()

    print(f"\nproduction projected_kl_loss:        {production.item():.10f}")
    print(f"independent re-derivation:           {independent_kl.item():.10f}")
    diff = abs(production.item() - independent_kl.item())
    print(f"absolute difference:                 {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit(
            f"production and independent computations disagree by {diff:.2e} — this "
            f"is the projection bug the exactness check exists to catch. Every X2 "
            f"result on this pair is suspect until this is resolved.")
    print("\nPASS: production X-Token loss matches an independently computed "
          "reference on real cached teacher data and a real student forward pass.")


if __name__ == "__main__":
    main()
