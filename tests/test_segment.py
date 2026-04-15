"""Tests for fixed-length waveform segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from music_classifier.preprocessing.segment import segment_waveform


SR = 22050


def _make_waveform(seconds: float, sr: int = SR) -> np.ndarray:
    return np.random.default_rng(0).uniform(-1.0, 1.0, int(seconds * sr)).astype(np.float32)


# ---------------------------------------------------------------------------
# Exact counts
# ---------------------------------------------------------------------------


def test_segment_count_divisible() -> None:
    """30 s / 3 s segments = exactly 10 non-overlapping segments."""
    y = _make_waveform(30.0)
    segs = segment_waveform(y, SR, segment_seconds=3.0)
    assert segs.shape == (10, int(3.0 * SR))


def test_segment_count_non_divisible_drop() -> None:
    """31 s / 3 s -> 10 full segments, last 1 s tail is dropped."""
    y = _make_waveform(31.0)
    segs = segment_waveform(y, SR, segment_seconds=3.0, pad_short=False)
    assert segs.shape[0] == 10


def test_segment_count_non_divisible_pad() -> None:
    """31 s / 3 s -> 11 segments when short tail is padded."""
    y = _make_waveform(31.0)
    segs = segment_waveform(y, SR, segment_seconds=3.0, pad_short=True)
    assert segs.shape == (11, int(3.0 * SR))


# ---------------------------------------------------------------------------
# Padding correctness
# ---------------------------------------------------------------------------


def test_padded_tail_zeros_at_end() -> None:
    """Tail segment after data should be zero-padded."""
    y = np.ones(int(3.5 * SR), dtype=np.float32)
    segs = segment_waveform(y, SR, segment_seconds=3.0, pad_short=True)
    tail = segs[1]  # second segment is the padded one
    n_real = int(0.5 * SR)
    assert np.all(tail[:n_real] == 1.0)
    assert np.all(tail[n_real:] == 0.0)


# ---------------------------------------------------------------------------
# Short clip (shorter than one segment)
# ---------------------------------------------------------------------------


def test_short_clip_drop_returns_empty() -> None:
    y = _make_waveform(1.0)
    segs = segment_waveform(y, SR, segment_seconds=3.0, pad_short=False)
    assert segs.shape == (0, int(3.0 * SR))


def test_short_clip_pad_returns_one_segment() -> None:
    y = _make_waveform(1.0)
    segs = segment_waveform(y, SR, segment_seconds=3.0, pad_short=True)
    assert segs.shape == (1, int(3.0 * SR))


# ---------------------------------------------------------------------------
# Overlapping hop
# ---------------------------------------------------------------------------


def test_overlapping_hop_count() -> None:
    """30 s with 3 s segment and 1.5 s hop -> (30 - 3) / 1.5 + 1 = 19 segments."""
    y = _make_waveform(30.0)
    segs = segment_waveform(y, SR, segment_seconds=3.0, hop_seconds=1.5)
    assert segs.shape[0] == 19


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------


def test_output_dtype_preserved() -> None:
    y = _make_waveform(10.0)
    assert y.dtype == np.float32
    segs = segment_waveform(y, SR, segment_seconds=3.0)
    assert segs.dtype == np.float32


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_2d_input_raises() -> None:
    y = np.zeros((2, SR), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        segment_waveform(y, SR, segment_seconds=1.0)


def test_nonpositive_segment_raises() -> None:
    y = _make_waveform(5.0)
    with pytest.raises(ValueError, match="segment_seconds"):
        segment_waveform(y, SR, segment_seconds=0.0)


def test_nonpositive_sr_raises() -> None:
    y = _make_waveform(5.0)
    with pytest.raises(ValueError, match="Sample rate"):
        segment_waveform(y, 0, segment_seconds=1.0)
