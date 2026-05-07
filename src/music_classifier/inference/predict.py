"""Batch preparation and prediction utilities.

This module prepares spectrogram data for model input and performs
batch prediction using a trained Keras model.

It ensures compatibility with Conv2D-based architectures by adding
the required channel dimension and enforcing proper memory layout.
"""

from __future__ import annotations

import numpy as np

from typing import Literal
from keras import Model

from .labels import GENRE_LABELS


def validate_prediction(predictions: np.ndarray) -> None:
    """Validate raw model prediction output.

    Ensures that prediction arrays returned by the model conform to the
    expected inference contract before aggregation and formatting.

    Validation checks include:
    1. Predictions must be a non-empty 2-D array.
    2. The class dimension must match ``GENRE_LABELS``.
    3. All confidence values must be within the range [0, 1].

    Parameters
    ----------
    predictions:
        2-D array of shape (n_segments, n_classes), where each row contains
        class probabilities for one audio segment.

    Raises
    ------
    ValueError
        If predictions are empty, not 2-D, contain an unexpected number
        of classes, or include confidence values outside the range [0, 1].
    """
    if predictions.ndim != 2:
        raise ValueError(
            "Predictions must be a 2-D array with shape "
            "(n_segments, n_classes)."
        )

    if predictions.size == 0:
        raise ValueError("No predictions were provided for aggregation.")

    if predictions.shape[1] != len(GENRE_LABELS):
        raise ValueError(
            "Model output shape does not match inference engine expectations."
        )

    if not np.all((predictions >= 0.0) & (predictions <= 1.0)):
        raise ValueError("Prediction confidences must be between 0 and 1.")


def format_prediction(predictions: np.ndarray) -> list[tuple[str, float]]:
    """Aggregate and format model predictions into ranked genre scores.

    This function validates raw model prediction output, aggregates
    segment-level predictions into a single probability distribution,
    and formats the results into ranked genre scores.

    Segment predictions are combined by computing the mean probability
    across all segments. Each probability is then paired with its
    corresponding genre label, sorted in descending order, and rounded
    to three decimal places for readability.

    Parameters
    ----------
    predictions:
        2-D array of shape (n_segments, n_classes), where each row contains
        class probabilities for one audio segment. ``n_classes`` must match
        the length of ``GENRE_LABELS``.

    Returns
    -------
    list[tuple[str, float]]
        Ranked list of (genre, probability) pairs sorted from highest to
        lowest probability.

    Raises
    ------
    ValueError
        If predictions are empty, not 2-D, contain an unexpected number
        of classes, or include confidence values outside the range [0, 1].
    """
    validate_prediction(predictions)
    aggregate = np.mean(predictions, axis=0)
    results = sorted(
        zip(GENRE_LABELS, aggregate, strict=False),
        key=lambda pair: pair[1],
        reverse=True
    )

    return [(label, round(score, 3)) for label, score in results]


def predict_batch(
    model: Model,
    spectrograms: np.ndarray,
    *,
    verbose: Literal["auto", 0, 1, 2] = 0
):
    """Run model inference and return formatted genre predictions.

    This function prepares spectrogram data for model input, performs
    batch prediction, and formats the results into ranked genre scores.

    Steps
    -----
    1. Validate spectrogram shape.
    2. Add a channel dimension for Conv2D input
    (n_segments, n_mels, n_frames) → (n_segments, n_mels, n_frames, 1).
    3. Run model prediction on all segments.
    4. Aggregate and format predictions via ``format_prediction``.

    Parameters
    ----------
    model:
        Trained Keras model.
    spectrograms:
        Array of spectrograms with shape (n_segments, n_mels, n_frames).
    verbose:
        Verbosity level for model prediction (default: 0).

    Returns
    -------
    list[tuple[str, float]]
        Ranked list of (genre, probability) pairs sorted from highest to
        lowest probability.

    Raises
    ------
    ValueError
        If the input array does not have 3 dimensions.
    """
    if spectrograms.ndim != 3:
        raise ValueError(
            f"Expected spectrograms with shape "
            f"(n_segments, n_mels, n_frames), got {spectrograms.shape}."
        )

    batch = spectrograms[..., np.newaxis]
    predictions = model.predict(batch, verbose=verbose)  # type: ignore
    results = format_prediction(predictions)

    return results
