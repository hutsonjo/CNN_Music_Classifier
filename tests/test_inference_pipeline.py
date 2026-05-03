from __future__ import annotations

from pathlib import Path

import numpy as np

from music_classifier.inference.pipeline import classify_file


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------


def test_classify_file_runs_pipeline_end_to_end(mocker) -> None:
    model = mocker.Mock()
    file_path = "song.wav"

    spectrograms = np.random.rand(5, 128, 130).astype(np.float32)
    spectrogram_record = {"spectrograms": spectrograms}
    expected = [("rock", 0.721), ("metal", 0.201)]

    build_batch_mock = mocker.patch(
        "music_classifier.inference.pipeline.build_batch",
        return_value=spectrogram_record,
    )
    predict_batch_mock = mocker.patch(
        "music_classifier.inference.pipeline.predict_batch",
        return_value=expected,
    )

    result = classify_file(model, file_path)

    build_batch_mock.assert_called_once_with(file_path)
    predict_batch_mock.assert_called_once_with(model, spectrograms)
    assert result == expected


def test_classify_file_returns_formatted_output_structure(mocker) -> None:
    model = mocker.Mock()
    file_path = Path("track.mp3")

    spectrogram_record = {
        "spectrograms": np.random.rand(5, 128, 130).astype(np.float32)
    }
    formatted_output = [("jazz", 0.555), ("blues", 0.333)]

    mocker.patch(
        "music_classifier.inference.pipeline.build_batch",
        return_value=spectrogram_record,
    )
    mocker.patch(
        "music_classifier.inference.pipeline.predict_batch",
        return_value=formatted_output,
    )

    result = classify_file(model, file_path)

    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)
    assert all(isinstance(label, str) for label, _ in result)
    assert all(isinstance(score, float) for _, score in result)
