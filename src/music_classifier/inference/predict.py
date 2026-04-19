from __future__ import annotations

from typing import Literal
import numpy as np
from tensorflow.keras import Model


def prepare_batch(spectrograms: np.ndarray) -> np.ndarray:
    if spectrograms.ndim != 3:
        raise ValueError(
            f"Expected spectrograms with shape (n_segments, n_mels, n_frames), got {spectrograms.shape}."
        )

    batch = spectrograms[..., np.newaxis]
    return np.ascontiguousarray(batch, dtype=np.float32)


def predict_batch(model: Model, spectrograms: np.ndarray, *, verbose: Literal[0] = 0) -> np.ndarray:
    batch = prepare_batch(spectrograms)
    predictions = model.predict(batch, verbose=verbose)
    return np.asarray(predictions, dtype=np.float32)