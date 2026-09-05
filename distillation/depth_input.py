"""Depth preprocessing for the student's visual input (plan §8.4, audit A2).

The student's depth input must be built from the **same decoding the gold answers
were derived from**. The legacy loader instead min-max normalized the stored PNG
integers per image, which:

* skips the official SUN RGB-D 16-bit rotation, so the low 3 bits are never moved
  to the top of the word. Measured on the released test split, that changes depth
  *ordering* on **20% of `xtion` frames** (up to 32% of pixels in the worst
  sampled frame) — and `xtion` is 34% of the test split;
* never applies the 8 m clip; and
* destroys metric scale, so an identical pixel value means a different distance in
  different images.

`decode_metric_depth` here is checked against `dataset_creation/v2/depth_utils.py`
— the decoder that produced the gold labels — on real frames from all four
sensors, so the two cannot drift apart silently.

Two student representations are provided, matching the plan's pilot table:

* ``"replicated"`` — decoded metric depth repeated over three channels.
* ``"gradient"``   — decoded depth, gradient magnitude, gradient orientation.

Both apply the *same* normalization and invalid-depth handling, so a comparison
between them is a comparison of representation and nothing else.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_CLIP_MAX_M = 8.0
REPRESENTATIONS = ("replicated", "gradient")

# Prewitt kernels, as used by the legacy gradient representation.
_KERNEL_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
_KERNEL_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


def decode_metric_depth(depth_path: str, clip_max_m: float = DEFAULT_CLIP_MAX_M) -> np.ndarray:
    """Decode a SUN RGB-D depth PNG to metres.

    Must stay byte-identical in behaviour to
    ``dataset_creation/v2/depth_utils.decode_sunrgbd_depth`` — the decoder behind
    every gold answer. `tests/test_depth_input.py` asserts that on real frames.
    """
    raw = np.array(Image.open(depth_path), dtype=np.uint16)
    return decode_raw_depth(raw, clip_max_m)


def decode_raw_depth(raw: np.ndarray, clip_max_m: float = DEFAULT_CLIP_MAX_M) -> np.ndarray:
    """The rotation itself, separated so it can be tested on synthetic values.

    ``(raw >> 3) | (raw << 13)`` moves the low 3 bits to the top of the uint16.
    For values that are multiples of 8 this reduces to a divide-by-8 and is
    order-preserving; for any other value it is not, which is the whole reason
    reading the PNG directly is wrong.
    """
    raw = raw.astype(np.uint16)
    rotated = (raw >> 3) | (raw << 13).astype(np.uint16)
    return np.clip(rotated.astype(np.float32) / 1000.0, 0.0, clip_max_m)


def valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    """Pixels carrying a real measurement. Zero means "no return" in SUN RGB-D."""
    return depth_m > 0.0


def scale_to_unit(depth_m: np.ndarray, clip_max_m: float = DEFAULT_CLIP_MAX_M) -> np.ndarray:
    """Map metres onto [0, 1] with a **fixed** ceiling, not a per-image one.

    A fixed divisor is what preserves metric comparability across images: the
    same pixel value means the same distance in every frame. Per-image min-max
    would make the encoding depend on whichever object happened to be nearest.
    """
    return np.clip(depth_m / clip_max_m, 0.0, 1.0)


def _prewitt(channel: np.ndarray):
    from scipy.ndimage import convolve
    gradient_x = convolve(channel, _KERNEL_X, mode="reflect")
    gradient_y = convolve(channel, _KERNEL_Y, mode="reflect")
    return gradient_x, gradient_y


def depth_to_student_input(depth_m: np.ndarray, representation: str = "replicated",
                           clip_max_m: float = DEFAULT_CLIP_MAX_M) -> np.ndarray:
    """Build the ``uint8`` HxWx3 array handed to the processor.

    Args:
        depth_m: decoded **metric** depth, as returned by :func:`decode_metric_depth`.
        representation: ``"replicated"`` or ``"gradient"``.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"unknown representation {representation!r}; expected one of {REPRESENTATIONS}")
    if depth_m.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {depth_m.shape}")

    scaled = scale_to_unit(depth_m, clip_max_m)
    depth_channel = (scaled * 255.0).astype(np.uint8)
    if representation == "replicated":
        return np.dstack([depth_channel] * 3)

    gradient_x, gradient_y = _prewitt(scaled.astype(np.float32))
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    orientation = np.arctan2(gradient_y, gradient_x)
    # Fixed ranges again, for the same reason as the depth channel: a per-image
    # rescale would make the encoding image-dependent. Magnitude is bounded by
    # 3*sqrt(2) for Prewitt on a [0, 1] input; orientation by [-pi, pi].
    magnitude_channel = (np.clip(magnitude / (3.0 * np.sqrt(2.0)), 0.0, 1.0) * 255.0).astype(np.uint8)
    orientation_channel = (((orientation + np.pi) / (2.0 * np.pi)) * 255.0).astype(np.uint8)
    return np.dstack([depth_channel, magnitude_channel, orientation_channel])


def depth_statistics(depth_m: np.ndarray) -> dict:
    """Diagnostics for the Phase 4 visual check and per-item reporting."""
    valid = valid_depth_mask(depth_m)
    valid_values = depth_m[valid]
    return {
        "valid_fraction": float(valid.mean()),
        "min_m": float(valid_values.min()) if valid_values.size else None,
        "median_m": float(np.median(valid_values)) if valid_values.size else None,
        "max_m": float(valid_values.max()) if valid_values.size else None,
        "clipped_fraction": float((depth_m >= DEFAULT_CLIP_MAX_M).mean()),
    }
