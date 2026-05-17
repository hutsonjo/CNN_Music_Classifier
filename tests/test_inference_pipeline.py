from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_classifier.inference import GENRE_LABELS, load_genre_model
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

    result = classify_file(model, file_path, top_n=None)

    build_batch_mock.assert_called_once_with(file_path)
    predict_batch_mock.assert_called_once_with(model, spectrograms)
    assert result == expected


def test_classify_file_returns_top_n_predictions(mocker) -> None:
    model = mocker.Mock()
    file_path = Path("track.mp3")

    spectrogram_record = {
        "spectrograms": np.random.rand(5, 128, 130).astype(np.float32)
    }
    formatted_output = [
        ("rock", 0.721),
        ("metal", 0.201),
        ("jazz", 0.050),
    ]

    mocker.patch(
        "music_classifier.inference.pipeline.build_batch",
        return_value=spectrogram_record,
    )
    mocker.patch(
        "music_classifier.inference.pipeline.predict_batch",
        return_value=formatted_output,
    )

    result = classify_file(model, file_path, 2)

    assert result == [
        ("rock", 0.721),
        ("metal", 0.201),
    ]


def test_classify_file_raises_when_top_n_is_less_than_one(mocker) -> None:
    model = mocker.Mock()
    file_path = Path("track.mp3")

    spectrogram_record = {
        "spectrograms": np.random.rand(5, 128, 130).astype(np.float32)
    }

    mocker.patch(
        "music_classifier.inference.pipeline.build_batch",
        return_value=spectrogram_record,
    )
    mocker.patch(
        "music_classifier.inference.pipeline.predict_batch",
        return_value=[("rock", 0.721)],
    )

    with pytest.raises(ValueError, match="top_n must be at least 1"):
        classify_file(model, file_path, 0)


def test_classify_file_raises_when_top_n_exceeds_genre_count(mocker) -> None:
    model = mocker.Mock()
    file_path = Path("track.mp3")

    spectrogram_record = {
        "spectrograms": np.random.rand(5, 128, 130).astype(np.float32)
    }

    mocker.patch(
        "music_classifier.inference.pipeline.build_batch",
        return_value=spectrogram_record,
    )
    mocker.patch(
        "music_classifier.inference.pipeline.predict_batch",
        return_value=[("rock", 0.721)],
    )

    with pytest.raises(ValueError, match="top_n cannot exceed"):
        classify_file(model, file_path, len(GENRE_LABELS) + 1)


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

    result = classify_file(model, file_path, top_n=None)

    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)
    assert all(isinstance(label, str) for label, _ in result)
    assert all(isinstance(score, float) for _, score in result)


def test_classify_file_uses_real_model_and_audio_file() -> None:
    model = load_genre_model()

    file_path = (
        Path(__file__).resolve().parents[1]
        / "sample_data"
        / "sample.mp3"
    )

    result = classify_file(model, file_path, 3)

    assert len(result) == 3

    assert all(
        label in GENRE_LABELS
        for label, _ in result
    )

    assert all(
        0.0 <= score <= 1.0
        for _, score in result
    )

    assert result[0][1] >= result[1][1]
    assert result[1][1] >= result[2][1]
