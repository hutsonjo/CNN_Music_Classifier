"""Integration tests against the real GTZAN dataset.

These tests require the dataset to be present at training_data/gtzan_dataset
and are marked with @pytest.mark.integration.

Run all tests including integration:
    pytest

Run only integration tests:
    pytest -m integration

Skip integration tests (unit tests only):
    pytest -m "not integration"
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from music_classifier.preprocessing.config import PreprocessConfig, SpectrogramConfig
from music_classifier.preprocessing.io import iter_audio_files, load_audio
from music_classifier.preprocessing.pipeline import (
    build_spectrogram_dataset,
    build_spectrogram_record,
    preprocess_dataset,
    preprocess_file,
)
from music_classifier.preprocessing.storage import load_dataset, save_dataset

GTZAN_ROOT = Path(__file__).parents[1] / "training_data" / "gtzan_dataset"

EXPECTED_GENRES = frozenset({
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
})
EXPECTED_FILES_PER_GENRE = 100
EXPECTED_TOTAL_FILES = len(EXPECTED_GENRES) * EXPECTED_FILES_PER_GENRE  # 1000

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not GTZAN_ROOT.is_dir(),
        reason=f"GTZAN dataset not found at {GTZAN_ROOT}",
    ),
]


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------


def test_all_ten_genres_present() -> None:
    """All 10 GTZAN genre folders must exist under the dataset root."""
    found = {p.name for p in GTZAN_ROOT.iterdir() if p.is_dir()}
    missing = EXPECTED_GENRES - found
    assert not missing, f"Missing genre folders: {missing}"


def test_file_count_per_genre() -> None:
    """Each genre folder must contain exactly 100 .au files."""
    counts = {
        genre_dir.name: sum(1 for f in genre_dir.iterdir() if f.suffix == ".au")
        for genre_dir in sorted(GTZAN_ROOT.iterdir())
        if genre_dir.is_dir() and genre_dir.name in EXPECTED_GENRES
    }
    wrong = {g: n for g, n in counts.items() if n != EXPECTED_FILES_PER_GENRE}
    assert not wrong, f"Unexpected file counts: {wrong}"


def test_discovery_yields_all_files() -> None:
    """iter_audio_files must yield exactly 1000 paths across all genres."""
    paths = list(iter_audio_files(GTZAN_ROOT))
    assert len(paths) == EXPECTED_TOTAL_FILES


def test_discovery_is_deterministic() -> None:
    """Two calls to iter_audio_files must return paths in identical order."""
    assert list(iter_audio_files(GTZAN_ROOT)) == list(iter_audio_files(GTZAN_ROOT))


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def test_au_loads_as_float32() -> None:
    """A real .au file must decode to a float32 mono waveform."""
    first = next(iter_audio_files(GTZAN_ROOT))
    y, sr = load_audio(first, target_sr=22050)
    assert sr == 22050
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.size > 0


def test_au_amplitude_in_valid_range() -> None:
    """Librosa normalises audio to [-1.0, 1.0] — values must stay in range."""
    first = next(iter_audio_files(GTZAN_ROOT))
    y, _ = load_audio(first, target_sr=22050)
    assert float(np.max(np.abs(y))) <= 1.0, "Amplitude exceeds normalised range"


def test_au_duration_roughly_30s() -> None:
    """GTZAN tracks are nominally 30 s — allow ±1 s tolerance."""
    first = next(iter_audio_files(GTZAN_ROOT))
    y, sr = load_audio(first, target_sr=22050)
    duration = len(y) / sr
    assert 29.0 <= duration <= 31.0, f"Unexpected duration: {duration:.2f}s"


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def test_preprocess_first_file_contract() -> None:
    """preprocess_file on a real .au must return a well-formed AudioRecord."""
    cfg = PreprocessConfig(target_sr=22050, segment_seconds=3.0)
    first = next(iter_audio_files(GTZAN_ROOT))
    record = preprocess_file(first, cfg)

    assert record["label"] in EXPECTED_GENRES
    assert record["sr"] == 22050
    assert record["segments"].dtype == np.float32
    assert record["segments"].ndim == 2
    assert record["segments"].shape[1] == int(3.0 * 22050)
    assert record["segments"].shape[0] > 0


def test_all_genres_represented_in_output() -> None:
    """After processing the full dataset, all 10 genre labels must appear."""
    cfg = PreprocessConfig(target_sr=22050, segment_seconds=3.0)
    labels = {r["label"] for r in preprocess_dataset(GTZAN_ROOT, cfg)}
    assert labels == EXPECTED_GENRES


def test_total_segment_count_in_expected_range() -> None:
    """Total segments must be between 9900 and 10000 (3 s windows, 30 s tracks).

    Exactly 10 000 would mean every track is a clean multiple of 3 s.
    The lower bound of 9900 gives generous room for tracks that are
    slightly short (observed: ~9991 with default pad_short=False).
    """
    cfg = PreprocessConfig(target_sr=22050, segment_seconds=3.0, pad_short=False)
    total = sum(r["segments"].shape[0] for r in preprocess_dataset(GTZAN_ROOT, cfg))
    assert 9900 <= total <= 10000, f"Unexpected total segment count: {total}"


def test_no_files_produce_zero_segments() -> None:
    """Every track must be long enough to yield at least one 3-second segment."""
    cfg = PreprocessConfig(target_sr=22050, segment_seconds=3.0, pad_short=False)
    empty = [
        str(r["path"]) for r in preprocess_dataset(GTZAN_ROOT, cfg)
        if r["segments"].shape[0] == 0
    ]
    assert not empty, f"Files with zero segments: {empty}"


def test_segment_counts_balanced_across_genres() -> None:
    """No genre should have dramatically fewer segments than the others.

    With 100 tracks of ~30 s each and 3 s windows, each genre should
    produce between 990 and 1000 segments.
    """
    cfg = PreprocessConfig(target_sr=22050, segment_seconds=3.0, pad_short=False)
    counts: Counter[str] = Counter()
    for record in preprocess_dataset(GTZAN_ROOT, cfg):
        counts[record["label"]] += record["segments"].shape[0]

    for genre, count in counts.items():
        assert 990 <= count <= 1000, (
            f"Genre '{genre}' has unexpected segment count: {count}"
        )


# ---------------------------------------------------------------------------
# Spectrogram generation (Stage 2) on real .au files
# ---------------------------------------------------------------------------

_PREPROCESS_CFG = PreprocessConfig(target_sr=22050, segment_seconds=3.0)
_SPECTROGRAM_CFG = SpectrogramConfig(n_mels=128, n_fft=2048, hop_length=512)


def test_spectrogram_shape_on_real_file() -> None:
    """Spectrogram output shape must be (n_segs, 128, ~130) on a real .au file."""
    first = next(iter_audio_files(GTZAN_ROOT))
    audio_record = preprocess_file(first, _PREPROCESS_CFG)
    spec_record = build_spectrogram_record(audio_record, _SPECTROGRAM_CFG)

    specs = spec_record["spectrograms"]
    assert specs.ndim == 3
    assert specs.shape[1] == 128  # n_mels
    # For 3 s at 22050 Hz with hop_length=512: floor(66150/512)+1 = 130
    assert 128 <= specs.shape[2] <= 132, f"Unexpected n_frames: {specs.shape[2]}"
    assert specs.shape[0] > 0


def test_spectrogram_dtype_on_real_file() -> None:
    """Normalised spectrograms must be float32."""
    first = next(iter_audio_files(GTZAN_ROOT))
    audio_record = preprocess_file(first, _PREPROCESS_CFG)
    spec_record = build_spectrogram_record(audio_record, _SPECTROGRAM_CFG)
    assert spec_record["spectrograms"].dtype == np.float32


def test_spectrogram_values_in_0_1_on_real_file() -> None:
    """minmax normalization must keep all values in [0, 1]."""
    first = next(iter_audio_files(GTZAN_ROOT))
    audio_record = preprocess_file(first, _PREPROCESS_CFG)
    spec_record = build_spectrogram_record(audio_record, _SPECTROGRAM_CFG)
    specs = spec_record["spectrograms"]
    assert float(specs.min()) >= 0.0
    assert float(specs.max()) <= 1.0


def test_full_pipeline_produces_saveable_dataset(tmp_path: Path) -> None:
    """Run the full two-stage pipeline on 5 files, save to .npz, reload and check."""
    cfg_p = PreprocessConfig(target_sr=22050, segment_seconds=3.0)
    cfg_s = SpectrogramConfig(n_mels=128, n_fft=2048, hop_length=512)

    records = []
    for i, spec_record in enumerate(build_spectrogram_dataset(GTZAN_ROOT, cfg_p, cfg_s)):
        records.append(spec_record)
        if i >= 4:  # process 5 files
            break

    assert len(records) == 5

    out = tmp_path / "test_dataset.npz"
    save_dataset(records, out)
    assert out.exists()

    X, y, label_names = load_dataset(out)

    total_segs = sum(r["spectrograms"].shape[0] for r in records)
    assert X.shape == (total_segs, 128, records[0]["spectrograms"].shape[2])
    assert y.shape == (total_segs,)
    assert len(label_names) > 0
    assert X.dtype == np.float32
    assert y.dtype == np.int64
