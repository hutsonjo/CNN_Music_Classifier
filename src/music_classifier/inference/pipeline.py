"""Inference pipeline: preprocessing → model prediction.

This module orchestrates the full inference workflow for a single audio
file. It connects the preprocessing pipeline to the trained model.

Steps
-----
1. Load and preprocess the audio file into waveform segments.
2. Convert segments into normalized mel-spectrograms.
3. Prepare spectrograms for model input.
4. Run prediction on each segment.

The current implementation returns raw prediction outputs. Aggregation
and label mapping will be added in future iterations.
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
from music_classifier.inference.predict import predict_batch


def build_batch(file_path: str | Path) -> SpectrogramRecord:
    """Generate spectrogram input from an audio file.

    This function runs the full preprocessing pipeline:
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
    spectrogram_record = build_spectrogram_record(audio_record, spectrogram_config)

    return spectrogram_record


def classify_file(
    model: Model,
    file_path: str | Path,
) -> list[tuple[str, float]]:
    """Run model inference on a single audio file.

    The audio file is first converted into spectrogram segments, then
    passed through the model to obtain formatted prediction probabilities.

    Parameters
    ----------
    model:
        Trained Keras model used for inference.
    file_path:
        Path to the audio file.

    Returns
    -------
    np.ndarray
        Array of shape (n_segments, n_classes) containing prediction
        probabilities for each segment.
    """
    spectrogram_record = build_batch(file_path)
    predictions = predict_batch(model, spectrogram_record['spectrograms'])
    
    return predictions
