"""Tests for .npz dataset serialisation and deserialisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_classifier.preprocessing.storage import load_dataset, save_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_MELS = 128
N_FRAMES = 130
GENRES = ["blues", "jazz", "rock"]


def _make_records(
    n_genres: int = 3,
    files_per_genre: int = 4,
    segs_per_file: int = 10,
) -> list[dict]:
    """Build minimal fake SpectrogramRecord dicts for storage testing."""
    records = []
    rng = np.random.default_rng(42)
    for genre in GENRES[:n_genres]:
        for i in range(files_per_genre):
            records.append({
                "path": Path(f"/data/{genre}/{genre}.{i:05d}.au"),
                "label": genre,
                "sr": 22050,
                "spectrograms": rng.random(
                    (segs_per_file, N_MELS, N_FRAMES), dtype=np.float64
                ).astype(np.float32),
            })
    return records


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------


def test_roundtrip_X_shape(tmp_path: Path) -> None:
    """X shape after load must equal (total_segments, n_mels, n_frames)."""
    n_genres, files_per_genre, segs = 3, 4, 10
    records = _make_records(n_genres, files_per_genre, segs)
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    X, y, label_names = load_dataset(out)
    assert X.shape == (n_genres * files_per_genre * segs, N_MELS, N_FRAMES)


def test_roundtrip_y_shape(tmp_path: Path) -> None:
    """y must be 1-D with one entry per segment."""
    records = _make_records(segs_per_file=10)
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    X, y, _ = load_dataset(out)
    assert y.ndim == 1
    assert y.shape[0] == X.shape[0]


def test_roundtrip_label_names(tmp_path: Path) -> None:
    """label_names must contain all genres that were saved."""
    records = _make_records(n_genres=3)
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    _, _, label_names = load_dataset(out)
    assert set(label_names) == set(GENRES[:3])


def test_roundtrip_label_encoding_consistent(tmp_path: Path) -> None:
    """label_names[y[i]] must equal the genre of segment i."""
    records = _make_records(n_genres=2, files_per_genre=2, segs_per_file=5)
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    X, y, label_names = load_dataset(out)

    # Reconstruct expected labels in save order
    expected_labels = []
    for r in records:
        expected_labels.extend([r["label"]] * r["spectrograms"].shape[0])

    decoded = [label_names[idx] for idx in y]
    assert decoded == expected_labels


def test_roundtrip_X_values_preserved(tmp_path: Path) -> None:
    """Spectrogram values must survive the save/load cycle unchanged."""
    records = _make_records(n_genres=1, files_per_genre=1, segs_per_file=3)
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    X, _, _ = load_dataset(out)
    expected = records[0]["spectrograms"]
    np.testing.assert_array_equal(X, expected)


def test_output_is_valid_npz(tmp_path: Path) -> None:
    """The output file must be loadable as a NumPy archive."""
    records = _make_records()
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    archive = np.load(out, allow_pickle=False)
    assert "X" in archive
    assert "y" in archive
    assert "label_names" in archive


def test_X_dtype_is_float32(tmp_path: Path) -> None:
    records = _make_records()
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    X, _, _ = load_dataset(out)
    assert X.dtype == np.float32


def test_y_dtype_is_int64(tmp_path: Path) -> None:
    records = _make_records()
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    _, y, _ = load_dataset(out)
    assert y.dtype == np.int64


def test_label_encoding_is_alphabetical(tmp_path: Path) -> None:
    """Genre labels must be integer-encoded in alphabetical order."""
    records = _make_records(n_genres=3)  # blues, jazz, rock
    out = tmp_path / "dataset.npz"
    save_dataset(records, out)
    _, _, label_names = load_dataset(out)
    assert label_names == sorted(label_names)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_save_empty_records_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_dataset([], tmp_path / "out.npz")


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "ghost.npz")
