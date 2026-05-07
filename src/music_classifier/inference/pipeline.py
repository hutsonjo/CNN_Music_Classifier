"""Inference pipeline: preprocessing → model prediction.

This module orchestrates the full inference workflow for a single audio
file. It connects the preprocessing pipeline to the trained model and
produces formatted genre predictions.

Steps
-----
1. Load and preprocess the audio file into waveform segments.
2. Convert segments into normalized mel-spectrograms.
3. Prepare spectrograms for model input.
4. Run model inference and aggregate predictions into ranked results.

This module provides a thin orchestration layer between preprocessing
and prediction utilities.
"""

from __future__ import annotations

from pathlib import Path
from keras import Model

from music_classifier.preprocessing import (
    PreprocessConfig,
    SpectrogramConfig,
    SpectrogramRecord,
    preprocess_file,
    build_spectrogram_record
)
from .predict import predict_batch
from .labels import GENRE_LABELS


def build_batch(file_path: str | Path) -> SpectrogramRecord:
    """Generate spectrogram input from an audio file.

    Runs the full preprocessing pipeline:
    load → segment → mel-spectrogram → normalize.

    Parameters
    ----------
    file_path:
        Path to the audio file.

    Returns
    -------
    SpectrogramRecord
        A record containing normalized spectrograms ready for inference.
    """
    path = Path(file_path)
    preprocess_config = PreprocessConfig()
    audio_record = preprocess_file(path, preprocess_config)

    spectrogram_config = SpectrogramConfig()
    spectrogram_record = build_spectrogram_record(
        audio_record,
        spectrogram_config
    )

    return spectrogram_record


def classify_file(
    model: Model,
    file_path: str | Path,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Run model inference on a single audio file.

    The audio file is converted into spectrogram segments and passed
    through the model to produce aggregated, ranked genre predictions.

    Parameters
    ----------
    model:
        Trained Keras model used for inference.
    file_path:
        Path to the audio file.
    top_n:
        Optional number of top predictions to return. If ``None``,
        all predictions are returned.

    Returns
    -------
    list[tuple[str, float]]
        Ranked list of (genre, probability) pairs, sorted from highest
        to lowest probability.

    Raises
    ------
    ValueError
        If ``top_n`` is less than 1 or exceeds the number of available
        genre labels.
    """
    spectrogram_record = build_batch(file_path)
    predictions = predict_batch(model, spectrogram_record['spectrograms'])

    if top_n is None:
        return predictions
    if top_n < 1:
        raise ValueError('top_n must be at least 1.')
    if top_n > len(GENRE_LABELS):
        raise ValueError(
            f"top_n cannot exceed the number of genres, "
            f"{len(GENRE_LABELS)}, got {top_n}"
        )

    predictions = predictions[:top_n]

    return predictions
