"""Pipeline orchestrator: load → resample → segment.

This module is the main entry point for the preprocessing step.  It wires
together the lower-level building blocks from ``io`` and ``segment`` so
callers only need to provide a file path (or dataset root) and a
``PreprocessConfig``; they get back ready-to-use ``AudioRecord`` dicts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np

from .config import PreprocessConfig, SpectrogramConfig
from .io import iter_audio_files, load_audio, parse_genre_label
from .normalize import normalize_spectrograms
from .segment import segment_waveform
from .spectrogram import segments_to_mel_spectrograms


class AudioRecord(TypedDict):
    """Output contract for a single preprocessed audio file.

    This is the data structure that flows out of the pipeline and into the
    training loop.  Every field has a stable name and type so downstream code
    can access results by key rather than by positional index.

    Fields
    ------
    path:
        Absolute path to the source audio file on disk.  Kept for traceability
        — useful when debugging a mislabelled or corrupted file.
    label:
        Genre name derived from the file's parent directory (e.g. ``"blues"``).
        This becomes the training target (the ground-truth class the model
        should learn to predict).
    sr:
        Sample rate of the preprocessed waveform in Hz.  Stored per-record so
        nothing upstream needs to remember what ``PreprocessConfig.target_sr``
        was set to.
    segments:
        2-D NumPy array of shape ``(n_segments, segment_samples)``.  Each row
        is one fixed-length audio clip ready to be converted into a
        spectrogram and fed to the model.  All rows within a record share the
        same sample rate and length.
    """

    path: Path
    label: str
    sr: int
    segments: np.ndarray  # shape (n_segments, segment_samples)


def preprocess_file(path: Path, config: PreprocessConfig) -> AudioRecord:
    """Run the full preprocessing pipeline on a single audio file.

    Executes the three steps in order:

    1. **Load** — decode the file and resample the waveform to
       ``config.target_sr`` using Librosa.
    2. **Segment** — chop the waveform into fixed-length clips according to
       ``config.segment_seconds``, ``config.hop_seconds``, and
       ``config.pad_short``.
    3. **Label** — read the genre from the file's parent directory name.

    Parameters
    ----------
    path:
        Path to the audio file.  Any format supported by soundfile/ffmpeg
        works (``*.au``, ``*.wav``, ``*.mp3``, etc.).
    config:
        Preprocessing hyperparameters.  See ``PreprocessConfig`` for details
        on each option.

    Returns
    -------
    AudioRecord
        A dict with keys ``path``, ``label``, ``sr``, and ``segments``.
        See ``AudioRecord`` for field descriptions.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist on disk.
    RuntimeError
        If the file exists but cannot be decoded (corrupt or unsupported
        format).
    ValueError
        If the decoded waveform is empty (zero-byte or silent-only file).
    """
    path = Path(path)
    y, sr = load_audio(path, target_sr=config.target_sr, mono=config.mono)
    segments = segment_waveform(
        y,
        sr,
        segment_seconds=config.segment_seconds,
        hop_seconds=config.hop_seconds,
        pad_short=config.pad_short,
    )
    return AudioRecord(
        path=path,
        label=parse_genre_label(path),
        sr=sr,
        segments=segments,
    )


def preprocess_dataset(
    dataset_root: Path,
    config: PreprocessConfig,
) -> Iterator[AudioRecord]:
    """Yield a preprocessed ``AudioRecord`` for every audio file in *dataset_root*.

    This is a *lazy generator*: it processes and yields one file at a time
    rather than loading the entire dataset into memory at once.  For 1 000 GTZAN
    files at 22 050 Hz × 30 s each, that would be ~2.5 GB of raw waveforms —
    streaming one file at a time keeps memory usage roughly constant regardless
    of dataset size.

    Files are visited in deterministic sorted order (genre folder → filename),
    which is critical for reproducible train/validation splits: if the split is
    "first 80 % of records = train", the same 800 files must land in the
    training set every time the pipeline is run.

    Errors on individual files are re-raised immediately.  The rationale is
    fail-fast: a silent skip could quietly corrupt the dataset split.  If you
    want to tolerate errors, wrap the call in a try/except at the call site and
    log the failure before continuing.

    Parameters
    ----------
    dataset_root:
        Root directory of the GTZAN-style dataset
        (e.g. ``training_data/gtzan_dataset``).  Must contain one subdirectory
        per genre, each holding the audio files for that genre.
    config:
        Preprocessing hyperparameters applied uniformly to every file.

    Yields
    ------
    AudioRecord
        One record per audio file, in sorted (genre, filename) order.
    """
    for audio_path in iter_audio_files(Path(dataset_root)):
        yield preprocess_file(audio_path, config)


class SpectrogramRecord(TypedDict):
    """Output contract for a single fully-preprocessed audio file.

    Extends ``AudioRecord`` by replacing raw waveform segments with normalised
    mel-spectrograms that are ready to pass directly into a CNN model.

    Fields
    ------
    path:
        Absolute path to the source audio file.  Kept for traceability.
    label:
        Genre name (e.g. ``"blues"``).  This is the training target — the
        ground-truth class the model should learn to predict.
    sr:
        Sample rate of the source waveform in Hz.
    spectrograms:
        3-D float32 array of shape ``(n_segments, n_mels, n_frames)``.
        Each slice ``spectrograms[i]`` is one normalised mel-spectrogram
        representing a fixed-length clip of the original audio.  Values
        are in the range [0, 1] for ``"minmax"`` normalization or
        approximately N(0, 1) for ``"standardize"``.
    """

    path: Path
    label: str
    sr: int
    spectrograms: np.ndarray  # (n_segments, n_mels, n_frames)


def build_spectrogram_record(
    audio_record: AudioRecord,
    config: SpectrogramConfig,
) -> SpectrogramRecord:
    """Convert a single ``AudioRecord`` into a ``SpectrogramRecord``.

    Applies mel-spectrogram generation followed by normalization to the
    waveform segments already stored in *audio_record*.

    Parameters
    ----------
    audio_record:
        Output of ``preprocess_file`` — contains raw waveform segments.
    config:
        Spectrogram hyperparameters (FFT size, mel bands, normalization
        strategy).  See ``SpectrogramConfig`` for full documentation.

    Returns
    -------
    SpectrogramRecord
        Same ``path``, ``label``, and ``sr`` as *audio_record*, with
        ``spectrograms`` replacing ``segments``.
    """
    raw_specs = segments_to_mel_spectrograms(
        audio_record["segments"],
        audio_record["sr"],
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmax=config.fmax,
    )
    norm_specs = normalize_spectrograms(raw_specs, strategy=config.normalize)
    return SpectrogramRecord(
        path=audio_record["path"],
        label=audio_record["label"],
        sr=audio_record["sr"],
        spectrograms=norm_specs,
    )


def build_spectrogram_dataset(
    dataset_root: Path,
    preprocess_config: PreprocessConfig,
    spectrogram_config: SpectrogramConfig,
) -> Iterator[SpectrogramRecord]:
    """Yield a ``SpectrogramRecord`` for every audio file under *dataset_root*.

    This is the top-level orchestrator for the full two-stage pipeline:

    1. **Stage 1** — ``preprocess_dataset``: load, resample, and segment each
       audio file into fixed-length waveform clips (``AudioRecord``).
    2. **Stage 2** — ``build_spectrogram_record``: convert each clip's
       waveform into a mel-spectrogram and normalise it (``SpectrogramRecord``).

    Like ``preprocess_dataset``, this is a *lazy generator* — it processes
    one file at a time so memory usage stays constant regardless of dataset
    size.  Files are yielded in deterministic sorted order.

    Parameters
    ----------
    dataset_root:
        Root directory of the GTZAN-style dataset.
    preprocess_config:
        Controls audio loading, resampling, and segmentation.
    spectrogram_config:
        Controls mel-spectrogram generation and normalization.

    Yields
    ------
    SpectrogramRecord
        One record per audio file, in sorted (genre, filename) order.
    """
    for audio_record in preprocess_dataset(Path(dataset_root), preprocess_config):
        yield build_spectrogram_record(audio_record, spectrogram_config)
