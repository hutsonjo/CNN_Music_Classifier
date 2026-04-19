"""Model loading utilities for inference.

This module is responsible for loading a trained Keras model from disk.
The model file must exist and contain both architecture and weights.

The returned model is ready for inference and can be used directly
with preprocessed spectrogram inputs.
"""

from __future__ import annotations

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def load_genre_model() -> Model:
    """Load the trained genre classification model from disk.

    Returns
    -------
    Model
        A Keras model loaded from the configured file path.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the expected location.
    """
    path = Path("model/model.keras")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)
