"""Tests for mel-spectrogram generation and per-spectrogram normalization."""

from __future__ import annotations

import numpy as np
import pytest

from music_classifier.preprocessing.normalize import normalize_spectrograms
from music_classifier.preprocessing.spectrogram import segments_to_mel_spectrograms

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_SECONDS = 3.0
N_SAMPLES = int(SEGMENT_SECONDS * SR)


def _make_segments(n: int = 4, n_samples: int = N_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, (n, n_samples)).astype(np.float32)


# ---------------------------------------------------------------------------
# segments_to_mel_spectrograms — shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape() -> None:
    """Output must be (n_segments, n_mels, n_frames) for known-length input."""
    segments = _make_segments(n=5)
    out = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )
    expected_frames = N_SAMPLES // HOP_LENGTH + 1
    assert out.shape == (5, N_MELS, expected_frames)


def test_output_dtype_is_float32() -> None:
    segments = _make_segments(n=2)
    out = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )
    assert out.dtype == np.float32


def test_output_values_are_finite() -> None:
    """dB-converted values must contain no NaN or Inf."""
    segments = _make_segments(n=4)
    out = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )
    assert np.all(np.isfinite(out)), "Spectrogram contains NaN or Inf values"


def test_output_values_are_non_positive_db() -> None:
    """librosa.power_to_db with ref=np.max anchors the peak at 0 dB.

    All values must therefore be <= 0 dB (energy at or below the loudest bin).
    """
    segments = _make_segments(n=3)
    out = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )
    assert float(out.max()) <= 0.0, "dB values must be non-positive (ref=np.max)"


def test_single_segment_shape() -> None:
    """A single-segment batch must return shape (1, n_mels, n_frames)."""
    segments = _make_segments(n=1)
    out = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )
    assert out.shape[0] == 1


def test_fmax_reduces_or_equals_default() -> None:
    """Setting fmax should not change the output shape."""
    segments = _make_segments(n=2)
    out_default = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )
    out_fmax = segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000.0
    )
    assert out_default.shape == out_fmax.shape


# ---------------------------------------------------------------------------
# segments_to_mel_spectrograms — validation
# ---------------------------------------------------------------------------


def test_1d_input_raises() -> None:
    """A 1-D waveform (not segmented) must raise ValueError."""
    y = np.zeros(N_SAMPLES, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        segments_to_mel_spectrograms(
            y, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
        )


def test_empty_segments_raises() -> None:
    segments = np.empty((0, N_SAMPLES), dtype=np.float32)
    with pytest.raises(ValueError, match="zero rows"):
        segments_to_mel_spectrograms(
            segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
        )


# ---------------------------------------------------------------------------
# normalize_spectrograms — minmax
# ---------------------------------------------------------------------------


def _make_spectrograms(n: int = 4) -> np.ndarray:
    """Return synthetic (n, n_mels, n_frames) float32 spectrograms."""
    segments = _make_segments(n=n)
    return segments_to_mel_spectrograms(
        segments, SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=None
    )


def test_minmax_output_in_0_1() -> None:
    specs = _make_spectrograms(n=6)
    normed = normalize_spectrograms(specs, strategy="minmax")
    assert float(normed.min()) >= 0.0
    assert float(normed.max()) <= 1.0


def test_minmax_dtype_is_float32() -> None:
    specs = _make_spectrograms(n=3)
    normed = normalize_spectrograms(specs, strategy="minmax")
    assert normed.dtype == np.float32


def test_minmax_shape_unchanged() -> None:
    specs = _make_spectrograms(n=5)
    normed = normalize_spectrograms(specs, strategy="minmax")
    assert normed.shape == specs.shape


def test_minmax_flat_spectrogram_returns_zeros() -> None:
    """A constant spectrogram (range = 0) must not produce NaN — return zeros."""
    specs = np.full((2, N_MELS, 10), fill_value=-40.0, dtype=np.float32)
    normed = normalize_spectrograms(specs, strategy="minmax")
    assert np.all(normed == 0.0)


# ---------------------------------------------------------------------------
# normalize_spectrograms — standardize
# ---------------------------------------------------------------------------


def test_standardize_mean_near_zero() -> None:
    specs = _make_spectrograms(n=4)
    normed = normalize_spectrograms(specs, strategy="standardize")
    for i in range(normed.shape[0]):
        assert abs(float(normed[i].mean())) < 1e-5, f"Mean not near zero for segment {i}"


def test_standardize_std_near_one() -> None:
    specs = _make_spectrograms(n=4)
    normed = normalize_spectrograms(specs, strategy="standardize")
    for i in range(normed.shape[0]):
        assert abs(float(normed[i].std()) - 1.0) < 1e-4, f"Std not near 1 for segment {i}"


def test_standardize_flat_returns_zeros() -> None:
    specs = np.full((2, N_MELS, 10), fill_value=-40.0, dtype=np.float32)
    normed = normalize_spectrograms(specs, strategy="standardize")
    assert np.all(normed == 0.0)


# ---------------------------------------------------------------------------
# normalize_spectrograms — validation
# ---------------------------------------------------------------------------


def test_normalize_2d_input_raises() -> None:
    bad = np.zeros((N_MELS, 130), dtype=np.float32)
    with pytest.raises(ValueError, match="3-D"):
        normalize_spectrograms(bad, strategy="minmax")


def test_normalize_unknown_strategy_raises() -> None:
    specs = _make_spectrograms(n=1)
    with pytest.raises(ValueError, match="strategy"):
        normalize_spectrograms(specs, strategy="unknown")  # type: ignore[arg-type]
