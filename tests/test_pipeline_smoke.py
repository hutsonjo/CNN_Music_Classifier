"""Smoke tests for the end-to-end preprocessing pipeline."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from music_classifier.preprocessing.config import PreprocessConfig
from music_classifier.preprocessing.pipeline import preprocess_dataset, preprocess_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050
SEGMENT_SECONDS = 3.0


def _write_wav(path: Path, duration_s: float = 30.0, sr: int = SR) -> Path:
    n_samples = int(duration_s * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return path


def _make_dataset(tmp_path: Path, n_genres: int = 2, n_files: int = 3) -> Path:
    """Create a minimal fake GTZAN-style dataset with WAV files."""
    genres = ["blues", "classical", "country", "disco", "hiphop"][:n_genres]
    for genre in genres:
        for i in range(n_files):
            _write_wav(tmp_path / genre / f"{genre}.{i:05d}.wav")
    return tmp_path


# ---------------------------------------------------------------------------
# preprocess_file
# ---------------------------------------------------------------------------


def test_preprocess_file_output_contract(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "blues" / "blues.00000.wav", duration_s=30.0)
    cfg = PreprocessConfig(target_sr=SR, segment_seconds=SEGMENT_SECONDS)
    record = preprocess_file(wav, cfg)

    assert record["path"] == wav
    assert record["label"] == "blues"
    assert record["sr"] == SR
    assert isinstance(record["segments"], np.ndarray)
    assert record["segments"].ndim == 2
    assert record["segments"].shape[1] == int(SEGMENT_SECONDS * SR)


def test_preprocess_file_correct_segment_count(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "rock" / "rock.00000.wav", duration_s=30.0)
    cfg = PreprocessConfig(target_sr=SR, segment_seconds=SEGMENT_SECONDS)
    record = preprocess_file(wav, cfg)
    assert record["segments"].shape[0] == 10


def test_preprocess_file_missing_raises(tmp_path: Path) -> None:
    cfg = PreprocessConfig()
    with pytest.raises(FileNotFoundError):
        preprocess_file(tmp_path / "ghost.wav", cfg)


# ---------------------------------------------------------------------------
# preprocess_dataset
# ---------------------------------------------------------------------------


def test_preprocess_dataset_yields_all_files(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path, n_genres=2, n_files=3)
    cfg = PreprocessConfig(target_sr=SR, segment_seconds=SEGMENT_SECONDS)
    records = list(preprocess_dataset(ds, cfg))
    assert len(records) == 6  # 2 genres × 3 files


def test_preprocess_dataset_sorted_order(tmp_path: Path) -> None:
    """Records must arrive in deterministic order (sorted by genre/filename)."""
    ds = _make_dataset(tmp_path, n_genres=2, n_files=3)
    cfg = PreprocessConfig(target_sr=SR, segment_seconds=SEGMENT_SECONDS)
    records = list(preprocess_dataset(ds, cfg))
    labels = [r["label"] for r in records]
    # blues comes before classical lexicographically
    assert labels[:3] == ["blues"] * 3
    assert labels[3:] == ["classical"] * 3


def test_preprocess_dataset_each_record_has_segments(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path, n_genres=2, n_files=2)
    cfg = PreprocessConfig(target_sr=SR, segment_seconds=SEGMENT_SECONDS)
    for record in preprocess_dataset(ds, cfg):
        assert record["segments"].shape[0] > 0, (
            f"Expected segments for {record['path']}"
        )


# Live dataset tests have moved to tests/test_gtzan_integration.py
