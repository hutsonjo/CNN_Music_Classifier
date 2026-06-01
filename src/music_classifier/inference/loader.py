"""Model loading utilities for inference.

This module is responsible for loading a trained Keras model from disk.
The model file must exist and contain both architecture and weights.

The returned model is ready for inference and can be used directly
with preprocessed spectrogram inputs. The model is loaded with
``compile=False`` to avoid requiring training-time configuration
during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from keras.models import load_model
from keras import Model
from .spec import SpecAugment


def load_genre_model() -> Model:
    """Load the trained genre classification model from disk.

    The model artifact is loaded from the configured project path and
    returned ready for inference. The model is loaded with
    ``compile=False`` so that training-time configuration is not required
    during inference.

    Returns
    -------
    Model
        A Keras model loaded from the configured file path.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the expected location.
    RuntimeError
        If the model fails to load.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "models"
        / "model.keras"
    )
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        return cast(Model, load_model(path, 
                                      compile=False,
                                      custom_objects={"SpecAugment": SpecAugment}))
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc
