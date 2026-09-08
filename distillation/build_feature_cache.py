"""Cache the teacher's pooled RGB vision features (signal kind `pooled_features`).

The feature-based rows (D0, D3, D5, D8) need the teacher's *intermediate*
representation, not its answer distribution, and `build_teacher_cache.py` writes
only `topk_logits`. Without this script `compose_loss` raises on every one of
them, so the entire feature half of the §9.2 ladder is unrunnable.

Three properties make this cache much cheaper than the logits cache:

**Only the vision tower runs.** A pooled vision feature is a function of the
image alone, so `model.visual(pixel_values, grid_thw)` is called directly and the
language model is never invoked. This is why the file is small and the run is
short despite loading a 9B teacher.

**The unit is the image, not the question.** The train split has 15,278 rows over
**4,187 distinct images** — caching per row would store the same vector 3.6 times
on average. Per-image is also the unit the contrastive negative bank needs, since
its exclusion rule is stated over scenes.

**It is prompt-independent.** No chat template is rendered and no text is
tokenised, so unlike the logits cache this one does not hash a prompt and is not
invalidated by a prompt change. `prompt_hash` is therefore recorded as null
deliberately — it is not an omission, and the §10.4 `<think>` defect that
invalidated every cached logit could not have affected a cache built this way.

The teacher reads **RGB** by default, which is the study's premise: the teacher
sees the modality the student does not have (§2). A depth-view cache is available
via `--modality` for the control rows that need it, and the modality is part of
the cache key, so the two can never be confused for one another.

Written as one consolidated `.npz` rather than one file per image (the convention
`build_teacher_cache.py` uses for its 15,278 rows). Two reasons: the whole table
is needed in memory at train time anyway, to sample negatives from; and the run
is short enough that resume machinery would cost more than an interrupted run
does. `--limit` exists to measure that claim before committing to a full split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.build_teacher_cache import dataset_version  # noqa: E402
from distillation.cache import CacheKey, cache_directory, write_cache_key  # noqa: E402
from distillation.features import pool_per_image  # noqa: E402
from distillation.train_student import build_image, load_rows  # noqa: E402

DEFAULT_CACHE_ROOT = os.path.join(PROJECT_ROOT, "checkpoints_scratch", "teacher_cache")

# Recorded into the cache key so a row's identity includes where its features
# came from. See `features.py` for why this layer and not the two alternatives.
FEATURE_LAYER = "visual.pooler_output(post_merger)"
CROP_AGGREGATION = "mean over merged tokens, L2-normalised, float32 accumulation"


def distinct_images(rows: list) -> list:
    """One row per distinct `image_id`, in first-appearance order.

    Deterministic on purpose: the row order fixes the feature table's row order,
    which fixes which rows a seeded negative-bank sample draws. A `set` here
    would make the bank unreproducible across runs for no benefit.
    """
    seen: dict = {}
    for row in rows:
        seen.setdefault(row["image_id"], row)
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--modality", default="rgb", choices=["rgb", "depth"],
                        help="The teacher's view. Default rgb — the study's premise "
                             "is that the teacher sees what the student cannot.")
    parser.add_argument("--representation", default="replicated",
                        help="Only used when --modality depth.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int,
                        help="First N distinct images only, to measure cost before "
                             "committing to a full split.")
    parser.add_argument("--out-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.teacher)
    config = AutoConfig.from_pretrained(args.teacher)
    teacher_revision = getattr(config, "_commit_hash", None)
    spatial_merge_size = config.vision_config.spatial_merge_size
    teacher_dim = config.vision_config.out_hidden_size

    transform = ("PIL RGB, processor default resize" if args.modality == "rgb"
                 else f"metric depth -> {args.representation}, processor default resize")
    key = CacheKey({
        "dataset_version": dataset_version(),
        "split": args.split,
        "teacher_model": args.teacher,
        "teacher_revision": teacher_revision,
        "processor_revision": teacher_revision,
        "precision": "bfloat16",
        # Deliberately null: this cache renders no prompt (see module docstring).
        "prompt_hash": None,
        ("rgb_transform" if args.modality == "rgb" else "depth_transform"): transform,
        "signal_kind": "pooled_features",
        "feature_layer": FEATURE_LAYER,
        "crop_aggregation": CROP_AGGREGATION,
    })
    directory = cache_directory(args.out_root, key)
    os.makedirs(directory, exist_ok=True)
    write_cache_key(directory, key)
    print(f"cache directory: {directory}", flush=True)
    print(json.dumps(key.describe(), indent=2), flush=True)

    all_rows = load_rows(args.split)
    images = distinct_images(all_rows)
    if args.limit:
        images = images[:args.limit]
    print(f"{len(images)} distinct images from {len(all_rows)} {args.split} rows; "
          f"teacher={args.teacher}; modality={args.modality}; dim={teacher_dim}",
          flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    features = np.zeros((len(images), teacher_dim), dtype=np.float16)
    image_ids, sequence_ids = [], []
    started = time.time()
    done = 0
    for start in range(0, len(images), args.batch_size):
        batch_rows = images[start:start + args.batch_size]
        pil = [build_image(row, args.modality, args.representation) for row in batch_rows]
        encoded = processor.image_processor(images=pil, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to("cuda:0", torch.bfloat16)
        grid_thw = encoded["image_grid_thw"].to("cuda:0")
        with torch.no_grad():
            # `pooler_output` is the post-merger sequence in language space —
            # verified identical to a forward hook on `visual.merger`. The
            # language model is never called.
            merged = model.model.visual(pixel_values, grid_thw=grid_thw).pooler_output
            pooled = pool_per_image(merged, grid_thw, spatial_merge_size, normalize=True)
        features[start:start + len(batch_rows)] = pooled.cpu().numpy().astype(np.float16)
        image_ids.extend(row["image_id"] for row in batch_rows)
        sequence_ids.extend(row["sequence_id"] for row in batch_rows)
        done += len(batch_rows)
        if done % args.log_every < args.batch_size:
            rate = done / (time.time() - started)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {done}/{len(images)} images  {rate:.2f} img/s  peak {peak:.2f} GB",
                  flush=True)

    path = os.path.join(directory, "feature_table.npz")
    np.savez_compressed(path, features=features,
                        image_ids=np.array(image_ids), sequence_ids=np.array(sequence_ids))

    elapsed = time.time() - started
    summary = {
        "cache_digest": key.digest(),
        "path": path,
        "split": args.split,
        "modality": args.modality,
        "images": len(images),
        "rows_covered": len(all_rows) if not args.limit else None,
        "distinct_sequences": len(set(sequence_ids)),
        "feature_dim": teacher_dim,
        "feature_layer": FEATURE_LAYER,
        "crop_aggregation": CROP_AGGREGATION,
        "dtype": "float16",
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "elapsed_minutes": round(elapsed / 60, 2),
        "images_per_second": round(len(images) / elapsed, 3) if elapsed else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "teacher_revision": teacher_revision,
    }
    with open(os.path.join(directory, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
