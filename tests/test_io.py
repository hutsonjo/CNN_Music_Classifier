"""Tests for audio file discovery, label parsing, and loading."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from music_classifier.preprocessing.io import (
    iter_audio_files,
    load_audio,
    parse_genre_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, duration_s: float = 1.0, sr: int = 22050) -> Path:
    """Write a minimal silent WAV file for testing (no external deps)."""
    n_samples = int(duration_s * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return path


# ---------------------------------------------------------------------------
# iter_audio_files
# ---------------------------------------------------------------------------


def test_iter_audio_files_sorted_deterministic(tmp_path: Path) -> None:
    """Discovery order must be identical across calls (sorted by folder/file)."""
    for genre in ["rock", "blues", "jazz"]:
        genre_dir = tmp_path / genre
        genre_dir.mkdir()
        for i in range(3):
            _write_wav(genre_dir / f"{genre}.{i:05d}.wav")

    paths_a = list(iter_audio_files(tmp_path))
    paths_b = list(iter_audio_files(tmp_path))
    assert paths_a == paths_b
    # First file should belong to 'blues' (lexicographically first genre)
    assert paths_a[0].parent.name == "blues"


def test_iter_audio_files_ignores_non_audio(tmp_path: Path) -> None:
    genre_dir = tmp_path / "blues"
    genre_dir.mkdir()
    _write_wav(genre_dir / "blues.00000.wav")
    (genre_dir / "README.txt").write_text("not audio")

    paths = list(iter_audio_files(tmp_path))
    assert all(p.suffix in {".wav", ".au", ".mp3", ".flac", ".ogg"} for p in paths)
    assert len(paths) == 1


def test_iter_audio_files_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list(iter_audio_files(tmp_path / "does_not_exist"))


# ---------------------------------------------------------------------------
# parse_genre_label
# ---------------------------------------------------------------------------


def test_parse_genre_label_returns_parent_name() -> None:
    p = Path("/some/dataset/blues/blues.00001.au")
    assert parse_genre_label(p) == "blues"


def test_parse_genre_label_all_gtzan_genres() -> None:
    genres = ["blues", "classical", "country", "disco", "hiphop",
              "jazz", "metal", "pop", "reggae", "rock"]
    for genre in genres:
        p = Path(f"/data/gtzan/{genre}/{genre}.00000.au")
        assert parse_genre_label(p) == genre


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


def test_load_audio_returns_float32_and_target_sr(tmp_path: Path) -> None:
    wav_path = _write_wav(tmp_path / "blues" / "test.wav", sr=44100)
    y, sr = load_audio(wav_path, target_sr=22050)
    assert sr == 22050
    assert y.dtype == np.float32


def test_load_audio_is_contiguous(tmp_path: Path) -> None:
    wav_path = _write_wav(tmp_path / "test.wav")
    y, _ = load_audio(wav_path, target_sr=22050)
    assert y.flags["C_CONTIGUOUS"]


def test_load_audio_mono_is_1d(tmp_path: Path) -> None:
    wav_path = _write_wav(tmp_path / "test.wav")
    y, _ = load_audio(wav_path, target_sr=22050, mono=True)
    assert y.ndim == 1


def test_load_audio_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio(tmp_path / "ghost.wav", target_sr=22050)
