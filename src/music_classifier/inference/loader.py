"""Model loading utilities for inference.

This module is responsible for loading a trained Keras model from disk.
The model file must exist and contain both architecture and weights.

The returned model is ready for inference and can be used directly
with preprocessed spectrogram inputs. The model is loaded with ``compile=False``
to avoid requiring training-time configuration during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from keras.models import load_model
from keras import Model


def load_genre_model() -> Model:
    """Load the trained genre classification model from disk.

    Returns
    -------
    Model
        A Keras model loaded from the configured file path.

    Raises
    ------
    NotImplementedError
        If the model artifact is not yet available.
    FileNotFoundError
        If the model file does not exist at the expected location.
    RuntimeError
        If the model fails to load.
    """
    path = Path(__file__).resolve().parents[2] / "model" / "model.keras"
    if not path.exists():
        raise NotImplementedError(
            f"Model artifact not available yet: expected at {path}"
        )
        # raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        return cast(Model, load_model(path, compile=False))
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc
