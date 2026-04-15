"""Audio file discovery, genre-label parsing, and Librosa-backed loading.

A *waveform* (also called a *signal*) is simply a 1-D array of numbers.  Each
number is the amplitude of the sound wave at one point in time — positive values
represent compressions of air, negative values represent rarefactions.  Librosa
reads audio files and hands these arrays back as NumPy arrays of float32 values
normalised to the range [-1.0, 1.0].
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import librosa
import numpy as np

_AUDIO_SUFFIXES: frozenset[str] = frozenset({".au", ".wav", ".mp3", ".flac", ".ogg"})


def iter_audio_files(dataset_root: Path) -> Iterator[Path]:
    """Yield all audio file paths under *dataset_root*, sorted deterministically.

    Expects the GTZAN layout::

        dataset_root/
          blues/
            blues.00000.au
            blues.00001.au
            ...
          classical/
            classical.00000.au
            ...

    The ``.au`` extension is the Sun/NeXT audio format used by GTZAN.  Librosa
    can read it natively.  Files with other common extensions (``.wav``,
    ``.mp3``, ``.flac``, ``.ogg``) are also included so the function works
    with converted datasets.

    Sorting by (genre_folder, filename) is essential for reproducibility: if
    files were yielded in filesystem order (which varies between operating
    systems and even between runs on the same machine), the train/validation
    split would differ each time, making experiment results impossible to
    compare.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")

    for genre_dir in sorted(dataset_root.iterdir()):
        if not genre_dir.is_dir():
            continue
        for audio_file in sorted(genre_dir.iterdir()):
            if audio_file.suffix.lower() in _AUDIO_SUFFIXES:
                yield audio_file


def parse_genre_label(path: Path) -> str:
    """Return the genre label for *path* by reading its parent directory name.

    GTZAN encodes the genre as the name of the folder containing the file, so
    ``gtzan/blues/blues.00000.au`` belongs to the ``"blues"`` class.  This
    function just returns that folder name — no parsing of the filename itself
    is needed.

    >>> parse_genre_label(Path("gtzan/blues/blues.00000.au"))
    'blues'
    """
    return Path(path).parent.name


def load_audio(
    path: Path,
    *,
    target_sr: int,
    mono: bool = True,
    dtype: type = np.float32,
) -> tuple[np.ndarray, int]:
    """Load an audio file from disk and return a normalised waveform array.

    Under the hood this calls ``librosa.load``, which decodes the file (via
    soundfile or ffmpeg), resamples to ``target_sr`` if necessary, and returns
    the audio as a NumPy array with values in the range [-1.0, 1.0].

    Parameters
    ----------
    path:
        Path to the audio file.  Any format understood by soundfile (WAV, FLAC,
        OGG, AU, …) or ffmpeg (MP3, AAC, …) is accepted.
    target_sr:
        The sample rate to resample to, in Hz.  *Resampling* means
        algorithmically stretching or compressing the waveform in time so that
        it contains exactly ``target_sr`` samples per second, regardless of the
        rate it was originally recorded at.  All files must share the same
        sample rate before they can be stacked into model batches.
    mono:
        When ``True``, a stereo or multi-channel recording is averaged down to
        a single channel before returning.  See ``PreprocessConfig.mono`` for
        the full rationale.
    dtype:
        NumPy scalar type for the returned array.  ``float32`` (the default)
        uses half the memory of ``float64`` with no meaningful loss of
        precision for audio; most deep-learning frameworks also prefer float32.

    Returns
    -------
    (y, sr)
        *y* — a contiguous NumPy array of shape ``(n_samples,)`` when
        ``mono=True``, or ``(n_channels, n_samples)`` for multi-channel output.
        Each element is the amplitude of the waveform at that sample, in the
        range [-1.0, 1.0].

        *sr* — the effective sample rate after loading (always equals
        ``target_sr``).  Returned alongside *y* so callers never have to track
        the sample rate separately.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    ValueError
        If the decoded waveform contains no samples (e.g. a zero-byte file).
    RuntimeError
        If Librosa or the underlying codec raises an unexpected decoding error,
        re-raised with the file path included for easier debugging.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=mono, dtype=dtype)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file {path}: {exc}") from exc

    if y.size == 0:
        raise ValueError(f"Loaded waveform is empty for file: {path}")

    # Ensure the array owns contiguous memory so downstream numpy/torch ops
    # never hit a silent copy-on-access from a non-contiguous view.
    y = np.ascontiguousarray(y)
    return y, sr
