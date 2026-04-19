"""Batch preparation and prediction utilities.

This module prepares spectrogram data for model input and performs
batch prediction using a trained Keras model.

It ensures compatibility with Conv2D-based architectures by adding
the required channel dimension and enforcing proper memory layout.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from tensorflow.keras import Model


def prepare_batch(spectrograms: np.ndarray) -> np.ndarray:
    """Convert spectrograms into model-ready batch format.

    Adds a channel dimension and ensures the array is contiguous
    and stored as float32 for efficient computation.

    Parameters
    ----------
    spectrograms:
        Array of shape (n_segments, n_mels, n_frames).

    Returns
    -------
    np.ndarray
        Array of shape (n_segments, n_mels, n_frames, 1).

    Raises
    ------
    ValueError
        If the input array does not have 3 dimensions.
    """
    if spectrograms.ndim != 3:
        raise ValueError(
            f"Expected spectrograms with shape (n_segments, n_mels, n_frames), got {spectrograms.shape}."
        )

    batch = spectrograms[..., np.newaxis]

    return np.ascontiguousarray(batch, dtype=np.float32)


def predict_batch(
    model: Model,
    spectrograms: np.ndarray,
    *,
    verbose: Literal[0] = 0
) -> np.ndarray:
    """Run model prediction on a batch of spectrograms.

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
    np.ndarray
        Prediction array of shape (n_segments, n_classes), where each
        row contains class probabilities for one segment.
    """
    batch = prepare_batch(spectrograms)
    predictions = model.predict(batch, verbose=verbose)

    return np.asarray(predictions, dtype=np.float32)
