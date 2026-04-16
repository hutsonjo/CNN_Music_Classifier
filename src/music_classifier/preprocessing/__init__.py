"""Audio preprocessing pipeline for the GTZAN music genre dataset.

This subpackage converts raw audio files into model-ready numpy arrays through
three sequential steps:

1. **Load** — read the audio file from disk and decode it into a waveform: a
   1-D array of floating-point numbers where each value represents the air
   pressure amplitude at one instant in time.

2. **Resample** — adjust the waveform so that every file has the same number of
   samples per second (``target_sr``).  Different recordings may have been
   captured at different rates; resampling normalises them so the model always
   sees the same time-to-sample relationship.

3. **Segment** — chop the (potentially long) waveform into fixed-length clips
   of ``segment_seconds`` each.  Neural networks require all inputs to be the
   same shape, and working with short clips also generates more training
   examples from a limited dataset.

Public API
----------
- ``PreprocessConfig`` — dataclass holding all hyperparameters for the pipeline.
- ``preprocess_file`` / ``preprocess_dataset`` — run the full pipeline on one
  file or an entire directory tree.
- ``iter_audio_files``, ``load_audio``, ``parse_genre_label``,
  ``segment_waveform`` — lower-level building blocks for custom workflows.
"""

from .config import PreprocessConfig
from .io import iter_audio_files, load_audio, parse_genre_label
from .pipeline import preprocess_dataset, preprocess_file
from .segment import segment_waveform

__all__ = [
    "PreprocessConfig",
    "iter_audio_files",
    "load_audio",
    "parse_genre_label",
    "preprocess_dataset",
    "preprocess_file",
    "segment_waveform",
]
