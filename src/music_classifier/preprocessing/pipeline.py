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

from .config import PreprocessConfig
from .io import iter_audio_files, load_audio, parse_genre_label
from .segment import segment_waveform


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
