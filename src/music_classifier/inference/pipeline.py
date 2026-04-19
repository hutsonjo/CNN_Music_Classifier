from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras import Model

from music_classifier.preprocessing import (
    PreprocessConfig,
    SpectrogramConfig,
    SpectrogramRecord,
    preprocess_file,
    build_spectrogram_record
)
from music_classifier.inference.predict import predict_batch


def build_batch(file_path: str | Path) -> SpectrogramRecord:
    path = Path(file_path)
    preprocess_config = PreprocessConfig()
    audio_record = preprocess_file(path, preprocess_config)
    spectrogram_config = SpectrogramConfig()
    spectrogram_record = build_spectrogram_record(audio_record, spectrogram_config)
    return spectrogram_record


def classify_file(
    model: Model,
    file_path: str | Path,
):
    spectrogram_record = build_batch(file_path)
    predictions = predict_batch(model, spectrogram_record['spectrograms'])

    if predictions.size == 0:
        raise ValueError("No predictions were provided for aggregation.")
    
    return predictions
