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
import keras
from keras import Model, layers
import tensorflow as tf

@keras.saving.register_keras_serializable()
class SpecAugment(layers.Layer):
    """
    SpecAugmentation that removes parts of frequency bands to
    help fight overfitting.

    When the model is loaded Keras needs to be able to access this custom 
    class as it is part of the model. However, this code does not run at 
    inference because Training is False. The inputs just run through. 
    """
    def __init__(self, freq_mask_param=15, time_mask_param=15, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, x, training=None):
        if not training:
            return x   # no-op at inference time

        shape = tf.shape(x)
        batch, time_steps, freq_bins, channels = shape[0], shape[1], shape[2], shape[3]

        # ── Frequency mask ───────────────────────────────────────────────────
        f  = tf.random.uniform((), 0, self.freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform((), 0, freq_bins - f,        dtype=tf.int32)
        freq_mask = tf.concat([
            tf.ones ([batch, time_steps, f0,             channels]),
            tf.zeros([batch, time_steps, f,              channels]),
            tf.ones ([batch, time_steps, freq_bins-f0-f, channels])
        ], axis=2)
        x = x * freq_mask

        # ── Time mask ────────────────────────────────────────────────────────
        t  = tf.random.uniform((), 0, self.time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform((), 0, time_steps - t,       dtype=tf.int32)
        time_mask = tf.concat([
            tf.ones ([batch, t0,              freq_bins, channels]),
            tf.zeros([batch, t,               freq_bins, channels]),
            tf.ones ([batch, time_steps-t0-t, freq_bins, channels])
        ], axis=1)
        x = x * time_mask

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"freq_mask_param": self.freq_mask_param,
                        "time_mask_param": self.time_mask_param})
        return config

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
        return cast(Model, load_model(path, compile=False))
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc
