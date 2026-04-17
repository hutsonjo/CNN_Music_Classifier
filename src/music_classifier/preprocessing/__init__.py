"""Audio preprocessing pipeline for the GTZAN music genre dataset.

This subpackage converts raw audio files into model-ready numpy arrays through
a two-stage pipeline:

**Stage 1 — Waveform preprocessing**

1. **Load** — read the audio file from disk and decode it into a waveform: a
   1-D array of floating-point numbers where each value represents the air
   pressure amplitude at one instant in time.
2. **Resample** — adjust the waveform so that every file has the same number
   of samples per second (``target_sr``).
3. **Segment** — chop the waveform into fixed-length clips of
   ``segment_seconds`` each.

**Stage 2 — Spectrogram generation**

4. **Mel-spectrogram** — convert each waveform clip into a 2-D image-like
   frequency-time representation using a mel filter bank and FFT.
5. **Normalize** — scale each spectrogram to a consistent numeric range
   ([0, 1] for ``"minmax"`` or N(0, 1) for ``"standardize"``).

**Utilities**

- **Stratified split** — divide source files into train/validation/test sets
  at the file level to prevent data leakage.
- **Storage** — save the fully processed dataset to a ``.npz`` archive and
  reload it as NumPy arrays ready for ``model.fit()``.

Public API
----------
Stage 1:
- ``PreprocessConfig`` — waveform preprocessing hyperparameters.
- ``preprocess_file`` / ``preprocess_dataset`` — run stage 1 on one file or a
  directory tree.

Stage 2:
- ``SpectrogramConfig`` — spectrogram generation and normalization hyperparameters.
- ``build_spectrogram_record`` / ``build_spectrogram_dataset`` — run stage 2.

Utilities:
- ``stratified_split`` — file-level stratified train/val/test split.
- ``save_dataset`` / ``load_dataset`` — persist and reload processed datasets.

Lower-level building blocks:
- ``iter_audio_files``, ``load_audio``, ``parse_genre_label``,
  ``segment_waveform``, ``segments_to_mel_spectrograms``,
  ``normalize_spectrograms``.
"""

from .config import PreprocessConfig, SpectrogramConfig
from .io import iter_audio_files, load_audio, parse_genre_label
from .normalize import normalize_spectrograms
from .pipeline import (
    AudioRecord,
    SpectrogramRecord,
    build_spectrogram_dataset,
    build_spectrogram_record,
    preprocess_dataset,
    preprocess_file,
)
from .segment import segment_waveform
from .spectrogram import segments_to_mel_spectrograms
from .splitter import stratified_split
from .storage import load_dataset, save_dataset

__all__ = [
    # Configs
    "PreprocessConfig",
    "SpectrogramConfig",
    # Records
    "AudioRecord",
    "SpectrogramRecord",
    # Stage 1
    "iter_audio_files",
    "load_audio",
    "parse_genre_label",
    "segment_waveform",
    "preprocess_file",
    "preprocess_dataset",
    # Stage 2
    "segments_to_mel_spectrograms",
    "normalize_spectrograms",
    "build_spectrogram_record",
    "build_spectrogram_dataset",
    # Utilities
    "stratified_split",
    "save_dataset",
    "load_dataset",
]
