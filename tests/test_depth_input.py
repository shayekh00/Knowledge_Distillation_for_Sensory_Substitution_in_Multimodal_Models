"""Depth decoding tests (plan §7.3, audit A2).

§7.3 requires that "depth decoding agrees with known encoded values and selected
outputs of the local SUN RGB-D toolbox". The corpus-backed tests skip cleanly when
the imagery is not unpacked, so the suite still runs on a bare checkout.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_creation", "v2")
for _path in (PROJECT_ROOT, _V2_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from depth_utils import decode_sunrgbd_depth  # noqa: E402

from distillation.depth_input import (  # noqa: E402
    DEFAULT_CLIP_MAX_M,
    decode_metric_depth,
    decode_raw_depth,
    depth_statistics,
    depth_to_student_input,
    scale_to_unit,
    valid_depth_mask,
)

RELEASE_TEST_CSV = os.path.join(PROJECT_ROOT, "release", "VQA-SUNRGBD-v2",
                                "rule_based", "test.csv")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


def sample_depth_paths(per_sensor=2):
    """One or two real depth frames per sensor, or [] when imagery is absent."""
    if not os.path.isfile(RELEASE_TEST_CSV):
        return []
    chosen, seen = [], {}
    with open(RELEASE_TEST_CSV) as handle:
        for row in csv.DictReader(handle):
            sensor = row["sensor"]
            if seen.get(sensor, 0) >= per_sensor:
                continue
            path = os.path.join(DATASET_DIR, row["depth_path"])
            if os.path.isfile(path):
                seen[sensor] = seen.get(sensor, 0) + 1
                chosen.append((sensor, path))
    return chosen


CORPUS = sample_depth_paths()
needs_corpus = pytest.mark.skipif(not CORPUS, reason="SUN RGB-D imagery not unpacked")


# ---------------------------------------------------------------------------
# Known encoded values
# ---------------------------------------------------------------------------

def test_rotation_matches_hand_computed_values():
    # Multiples of 8 have empty low 3 bits, so the rotation is a divide-by-8.
    raw = np.array([[8000, 16000, 800]], dtype=np.uint16)
    assert np.allclose(decode_raw_depth(raw), [[1.0, 2.0, 0.1]])


def test_low_three_bits_wrap_to_the_top_of_the_word():
    """This is the case a naive `raw / 1000` reading gets wrong."""
    # 8001 = 8000 + 1: the trailing 1 rotates to bit 13 -> +8192, then /1000.
    decoded = decode_raw_depth(np.array([[8001]], dtype=np.uint16), clip_max_m=100.0)
    assert decoded[0, 0] == pytest.approx((8001 >> 3 | (8001 << 13) & 0xFFFF) / 1000.0)
    # And it is emphatically not order-preserving against the raw value.
    assert decoded[0, 0] > decode_raw_depth(np.array([[16000]], dtype=np.uint16),
                                            clip_max_m=100.0)[0, 0]


def test_clip_is_applied_at_eight_metres():
    raw = np.array([[8000 * 8]], dtype=np.uint16)     # decodes far beyond 8 m
    assert decode_raw_depth(raw).max() <= DEFAULT_CLIP_MAX_M


def test_zero_means_no_return():
    assert decode_raw_depth(np.zeros((2, 2), dtype=np.uint16)).sum() == 0.0
    assert not valid_depth_mask(np.zeros((2, 2), dtype=np.float32)).any()


# ---------------------------------------------------------------------------
# Agreement with the decoder that produced the gold answers
# ---------------------------------------------------------------------------

@needs_corpus
def test_agrees_with_the_v2_generator_decoder_on_every_sensor():
    for sensor, path in CORPUS:
        ours = decode_metric_depth(path)
        theirs = decode_sunrgbd_depth(path, DEFAULT_CLIP_MAX_M)
        assert np.array_equal(ours, theirs), f"decoder drift on {sensor}: {path}"


@needs_corpus
def test_metric_scale_is_shared_across_images():
    """Per-image min-max would make the same value mean different distances."""
    encodings = []
    for _sensor, path in CORPUS[:4]:
        depth = decode_metric_depth(path)
        channel = depth_to_student_input(depth, "replicated")[:, :, 0]
        valid = valid_depth_mask(depth)
        if valid.any():
            # A fixed 8 m ceiling means value/255*8 recovers metres in any image.
            recovered = channel[valid].astype(np.float32) / 255.0 * DEFAULT_CLIP_MAX_M
            encodings.append(np.abs(recovered - depth[valid]).max())
    assert encodings and max(encodings) < 8.0 / 255.0 + 1e-6


@needs_corpus
def test_statistics_are_reported_for_real_frames():
    _sensor, path = CORPUS[0]
    stats = depth_statistics(decode_metric_depth(path))
    assert 0.0 <= stats["valid_fraction"] <= 1.0
    assert stats["median_m"] is None or 0.0 < stats["median_m"] <= DEFAULT_CLIP_MAX_M


# ---------------------------------------------------------------------------
# Student input construction
# ---------------------------------------------------------------------------

def synthetic_depth():
    return np.linspace(0.0, DEFAULT_CLIP_MAX_M, 64, dtype=np.float32).reshape(8, 8)


def test_replicated_representation_has_three_identical_channels():
    image = depth_to_student_input(synthetic_depth(), "replicated")
    assert image.shape == (8, 8, 3) and image.dtype == np.uint8
    assert np.array_equal(image[:, :, 0], image[:, :, 1])
    assert np.array_equal(image[:, :, 1], image[:, :, 2])


def test_gradient_representation_shares_its_depth_channel_with_replicated():
    """Same normalization in both, so a comparison isolates the representation."""
    depth = synthetic_depth()
    replicated = depth_to_student_input(depth, "replicated")
    gradient = depth_to_student_input(depth, "gradient")
    assert gradient.shape == (8, 8, 3)
    assert np.array_equal(replicated[:, :, 0], gradient[:, :, 0])


def test_scaling_uses_a_fixed_ceiling_not_the_image_range():
    near = np.full((4, 4), 1.0, dtype=np.float32)
    far = np.full((4, 4), 4.0, dtype=np.float32)
    # Under per-image min-max both would map to the same constant; they must not.
    assert scale_to_unit(near).max() == pytest.approx(0.125)
    assert scale_to_unit(far).max() == pytest.approx(0.5)


def test_construction_is_deterministic():
    depth = synthetic_depth()
    for representation in ("replicated", "gradient"):
        first = depth_to_student_input(depth, representation)
        second = depth_to_student_input(depth, representation)
        assert np.array_equal(first, second)


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="unknown representation"):
        depth_to_student_input(synthetic_depth(), "rgb")


def test_non_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError, match="2-D depth map"):
        depth_to_student_input(np.zeros((4, 4, 3), dtype=np.float32))
