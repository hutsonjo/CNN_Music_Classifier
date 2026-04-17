"""Tests for stratified file-level train/val/test splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_classifier.preprocessing.splitter import stratified_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


def _make_records(
    n_genres: int = 10,
    files_per_genre: int = 100,
) -> list[dict]:
    """Build minimal fake SpectrogramRecord dicts for split testing."""
    records = []
    for genre in GENRES[:n_genres]:
        for i in range(files_per_genre):
            records.append({
                "path": Path(f"/data/{genre}/{genre}.{i:05d}.au"),
                "label": genre,
                "sr": 22050,
                "spectrograms": np.zeros((10, 128, 130), dtype=np.float32),
            })
    return records


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_all_records_in_exactly_one_split() -> None:
    """Every input record must appear in exactly one of the three splits."""
    records = _make_records(n_genres=5, files_per_genre=20)
    train, val, test = stratified_split(records, seed=42)

    all_paths = [str(r["path"]) for r in records]
    seen_paths = (
        [str(r["path"]) for r in train]
        + [str(r["path"]) for r in val]
        + [str(r["path"]) for r in test]
    )
    assert sorted(seen_paths) == sorted(all_paths), "Records missing or duplicated"


def test_no_path_appears_in_two_splits() -> None:
    """File-level split: no source file may appear in more than one subset."""
    records = _make_records(n_genres=5, files_per_genre=20)
    train, val, test = stratified_split(records, seed=42)

    train_paths = {str(r["path"]) for r in train}
    val_paths = {str(r["path"]) for r in val}
    test_paths = {str(r["path"]) for r in test}

    assert not train_paths & val_paths, "Overlap between train and val"
    assert not train_paths & test_paths, "Overlap between train and test"
    assert not val_paths & test_paths, "Overlap between val and test"


def test_split_ratios_approximately_correct() -> None:
    """Each split should be within ±1 file per genre of the requested ratio."""
    n_files = 100
    records = _make_records(n_genres=10, files_per_genre=n_files)
    train, val, test = stratified_split(
        records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    total = len(records)
    assert abs(len(train) / total - 0.8) < 0.02
    assert abs(len(val) / total - 0.1) < 0.02
    assert abs(len(test) / total - 0.1) < 0.02


def test_all_genres_in_each_split() -> None:
    """Every genre must be represented in all three splits."""
    records = _make_records(n_genres=10, files_per_genre=20)
    train, val, test = stratified_split(records, seed=42)

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        genres_in_split = {r["label"] for r in split}
        missing = set(GENRES) - genres_in_split
        assert not missing, f"Genres missing from {split_name}: {missing}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_split() -> None:
    records = _make_records(n_genres=5, files_per_genre=20)
    train_a, val_a, test_a = stratified_split(records, seed=7)
    train_b, val_b, test_b = stratified_split(records, seed=7)

    assert [str(r["path"]) for r in train_a] == [str(r["path"]) for r in train_b]
    assert [str(r["path"]) for r in val_a] == [str(r["path"]) for r in val_b]
    assert [str(r["path"]) for r in test_a] == [str(r["path"]) for r in test_b]


def test_different_seeds_produce_different_splits() -> None:
    records = _make_records(n_genres=5, files_per_genre=20)
    train_a, _, _ = stratified_split(records, seed=1)
    train_b, _, _ = stratified_split(records, seed=2)

    paths_a = [str(r["path"]) for r in train_a]
    paths_b = [str(r["path"]) for r in train_b]
    assert paths_a != paths_b, "Different seeds produced identical splits"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        stratified_split([])


def test_ratios_not_summing_to_one_raises() -> None:
    records = _make_records(n_genres=2, files_per_genre=10)
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(records, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


def test_negative_ratio_raises() -> None:
    records = _make_records(n_genres=2, files_per_genre=10)
    with pytest.raises(ValueError, match="non-negative"):
        stratified_split(records, train_ratio=0.9, val_ratio=0.2, test_ratio=-0.1)
