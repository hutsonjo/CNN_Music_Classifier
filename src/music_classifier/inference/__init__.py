"""Inference package public API.

This module exposes the high-level functions used for model inference:

- ``load_genre_model`` — load a trained Keras model from disk.
- ``build_batch`` — use preprocessing pipeline to convert audio file to spectrogram
- ``classify_file`` — run end-to-end prediction on a single audio file.
- ``

These functions form the boundary between the CLI layer and the
underlying inference implementation.
"""

from.labels import GENRE_LABELS
from .loader import load_genre_model
from .pipeline import build_batch, classify_file
from .predict import format_prediction, predict_batch

__all__ = [
    "GENRE_LABELS",
    "load_genre_model",
    "build_batch",
    "classify_file",
    "format_prediction",
    "predict_batch"
]
