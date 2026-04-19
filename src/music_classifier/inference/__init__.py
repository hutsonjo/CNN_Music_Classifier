"""High-level inference API for the music genre classifier.

This package defines the public interface used to perform inference with a
trained model. It acts as the boundary between user-facing layers (such as
the CLI) and the underlying preprocessing and model logic.

Overview
--------
The inference workflow consists of three main stages:

1. Model loading
   - ``load_genre_model()`` loads a trained Keras model from disk.

2. Input preparation
   - ``build_batch()`` converts an audio file into a batch of normalized
     mel-spectrograms using the preprocessing pipeline.

3. Prediction
   - ``predict_batch()`` runs the model on spectrogram data and produces
     formatted, ranked genre predictions.
   - ``classify_file()`` is a convenience wrapper that performs the full
     end-to-end process for a single audio file.

Public API
----------
GENRE_LABELS
    Ordered list of genre names corresponding to model output indices.

load_genre_model() -> keras.Model
    Load the trained genre classification model from disk.

build_batch(file_path) -> SpectrogramRecord
    Convert an audio file into model-ready spectrogram data.

predict_batch(model, spectrograms) -> list[tuple[str, float]]
    Run inference on a batch of spectrograms and return ranked predictions.

format_prediction(predictions) -> list[tuple[str, float]]
    Aggregate segment-level predictions into a single ranked result.

classify_file(model, file_path) -> list[tuple[str, float]]
    End-to-end inference for a single audio file.
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
