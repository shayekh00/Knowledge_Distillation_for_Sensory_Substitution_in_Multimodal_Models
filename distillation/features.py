"""Pooled vision-feature alignment: the declared feature layer, and the pieces
stage F needs that are pure functions of tensors.

**The declared feature layer (§9.3, amended 2026-09-07): the output of
`model.visual`, i.e. `pooler_output`, which is the post-merger vision sequence
in language-model space.** Three candidates existed and the choice is recorded
because §12 makes the feature layer part of a row's identity:

* `model.visual` `last_hidden_state` — pre-merger patch tokens, vision width
  (student 768, teacher 1152). This is the closest analogue of the legacy
  implementation's `vision_tower.vision_model.post_layernorm` hook.
* `model.visual` `pooler_output` — post-merger, language width (student 1024,
  teacher 4096). **Chosen.**
* A language-model hidden state — rejected: it mixes the question's text tokens
  into the representation, so a "vision feature" loss would partly align
  wording, and the teacher's cache would stop being prompt-independent.

`pooler_output` wins on two grounds. It is the representation the language model
actually consumes, so aligning it targets what reaches the answer rather than an
internal vision activation the merger may discard. And its width equals each
model's text hidden size, which makes the alignment map a transform between two
well-defined language spaces instead of between two arbitrary vision widths.

Qwen3.5 has no `post_layernorm`, so the legacy hook name does not exist here;
`pooler_output` is verified equal to a forward hook on `model.visual.merger`
(`test_pooler_output_is_the_merger_output`, and confirmed on the real 0.8B).

**Pooling is a mean over each image's own merged tokens, then L2 normalisation.**
The merged sequence arrives flat — `[sum_i tokens_i, D]` for a whole batch, not
`[B, T, D]` — because Qwen packs every image's patches into one axis. Splitting
it needs `image_grid_thw`, and getting that split wrong is silent: the shapes
still work and the features are simply wrong, mixing one image's patches into
another's mean. `merged_token_counts` is therefore checked against the tensor it
is about to split, and disagreement raises.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def merged_token_counts(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Merged-token count per image, from the processor's patch grid.

    `grid_thw` rows are `(t, h, w)` in *patch* units; the merger pools each
    `spatial_merge_size x spatial_merge_size` neighbourhood, so an image's token
    count divides by the square of it. Qwen's grids are always even in h and w,
    but an odd grid would make this floor-divide silently drop a row of patches,
    so it is rejected rather than truncated.
    """
    if grid_thw.dim() != 2 or grid_thw.size(-1) != 3:
        raise ValueError(f"grid_thw must be [B, 3], got {tuple(grid_thw.shape)}")
    if spatial_merge_size < 1:
        raise ValueError(f"spatial_merge_size must be >= 1, got {spatial_merge_size}")
    products = grid_thw.prod(-1)
    block = spatial_merge_size * spatial_merge_size
    remainder = products % block
    if bool((remainder != 0).any()):
        bad = int((remainder != 0).nonzero()[0])
        raise ValueError(
            f"image {bad} has patch grid {grid_thw[bad].tolist()}, whose product is not "
            f"divisible by spatial_merge_size^2 = {block}. Floor division here would "
            "silently drop patches from that image's pooled feature.")
    return products // block


def pool_per_image(merged: torch.Tensor, grid_thw: torch.Tensor,
                   spatial_merge_size: int, normalize: bool = True) -> torch.Tensor:
    """`[total_tokens, D]` packed merger output -> `[B, D]`, one vector per image.

    The mean is taken in float32 even when the model runs bf16: an image
    contributes a few hundred merged tokens, and bf16 has ~3 decimal digits, so
    accumulating the sum in bf16 loses real precision in a quantity that then
    gets L2-normalised and compared by cosine.

    Differentiable — this is the student's path, and stage F's gradient reaches
    the vision blocks through it.
    """
    if merged.dim() != 2:
        raise ValueError(f"merged features must be [total_tokens, D], got {tuple(merged.shape)}")
    counts = merged_token_counts(grid_thw, spatial_merge_size)
    total = int(counts.sum())
    if total != merged.size(0):
        raise ValueError(
            f"grid_thw accounts for {total} merged tokens but the feature tensor has "
            f"{merged.size(0)}. Splitting on a stale or mismatched grid would mix one "
            "image's patches into another's pooled feature without any shape error.")
    segments = merged.float().split(counts.tolist())
    pooled = torch.stack([segment.mean(dim=0) for segment in segments])
    return F.normalize(pooled, dim=-1) if normalize else pooled


class AlignmentHead(nn.Module):
    """Maps the student's pooled feature into the teacher's feature width.

    Needed because `feature_transfer_loss` and `contrastive_loss` both require
    equal widths, and this pair does not have them: student 1024, teacher 4096.

    The transform is applied to the **student** side only. The teacher's cached
    features are the target and are never transformed — a learnable map on the
    target side could reduce the loss by degrading the target, which is not
    alignment. This is also why the head is deliberately a single bias-free
    linear map: it is the weakest thing that can fix a width mismatch, so a gain
    is less easily attributed to extra capacity smuggled in beside the student.

    Distinct from the model's own `visual.merger`, which §8.2 calls "the
    projector" and freezes in stage F. This head is not part of the pretrained
    model and does not survive into deployment: it exists only to compute the
    stage-F loss, and §6's deployable student never runs it.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim < 1 or teacher_dim < 1:
            raise ValueError(f"dims must be positive, got {student_dim} -> {teacher_dim}")
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.project = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.size(-1) != self.student_dim:
            raise ValueError(
                f"expected student features of width {self.student_dim}, got {pooled.size(-1)}")
        return F.normalize(self.project(pooled), dim=-1)


def sample_negative_bank(features: torch.Tensor, sequence_ids: list[str],
                         exclude_sequences: set, size: int,
                         generator: torch.Generator | None = None) -> tuple:
    """`size` negatives from `size` **distinct** scenes, excluding a batch's own.

    Implements the audit's B4 requirement (`implementation_audit.md`): 255
    negatives plus the one positive, "sampled from distinct training scenes
    excluding room/sequence neighbours", with the distinct-scene count logged and
    a one-scene configuration made a hard error. The historical bug this replaces
    built an `[B, B]` similarity matrix from the physical batch, which at batch 1
    is a `[1, 1]` cross-entropy that is *identically zero* — the contrastive term
    silently contributed nothing.

    Excluding by **sequence** rather than by image matters here: 109 of this
    split's 3,231 sequences hold more than one image (one holds 52), so a
    different image of the same room is a near-duplicate of the positive and
    would be a false negative.

    Returns `(bank, distinct_scenes)`. `bank` is `[n, D]` where `n <= size` if
    fewer eligible scenes exist than requested.
    """
    if features.dim() != 2:
        raise ValueError(f"feature table must be [S, D], got {tuple(features.shape)}")
    if len(sequence_ids) != features.size(0):
        raise ValueError(
            f"{len(sequence_ids)} sequence ids for {features.size(0)} feature rows")
    if size < 1:
        raise ValueError(f"bank size must be >= 1, got {size}")

    # One representative row per eligible scene, so the sample is over scenes.
    first_row_of_scene: dict = {}
    for row, sequence in enumerate(sequence_ids):
        if sequence in exclude_sequences:
            continue
        first_row_of_scene.setdefault(sequence, row)
    if not first_row_of_scene:
        raise ValueError(
            "no scene remains after excluding the batch's own sequences, so every "
            "candidate would be a room neighbour of its own positive. A contrastive "
            "loss cannot be computed from a single scene (audit B4): it is identically "
            "zero and produces no gradient.")

    rows = torch.tensor(sorted(first_row_of_scene.values()), dtype=torch.long)
    take = min(size, rows.numel())
    chosen = rows[torch.randperm(rows.numel(), generator=generator)[:take]]
    return features[chosen], int(take)
