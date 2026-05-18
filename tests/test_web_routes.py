"""Tests for Flask web API routes and upload handling."""

from __future__ import annotations

import io
import wave

from music_classifier.web.app import create_app


def make_wav_file() -> io.BytesIO:
    """Create a small in-memory WAV file for upload tests."""

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x00" * 22050)

    buffer.seek(0)
    return buffer


def test_predict_uploads_audio_file(mocker):
    """POST /predict accepts an uploaded WAV file and returns predictions."""

    fake_model = mocker.Mock()

    classify_file_mock = mocker.patch(
        "music_classifier.web.routes.classify_file",
        return_value=[
            ("rock", 0.721),
            ("metal", 0.201),
            ("jazz", 0.05),
        ],
    )

    app = create_app(model=fake_model)
    client = app.test_client()

    response = client.post(
        "/predict",
        data={
            "file": (make_wav_file(), "sample.wav"),
            "top_n": "3",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "predictions": [
            {"genre": "rock", "confidence": 0.721},
            {"genre": "metal", "confidence": 0.201},
            {"genre": "jazz", "confidence": 0.05},
        ]
    }

    classify_file_mock.assert_called_once()
    _, kwargs = classify_file_mock.call_args

    assert kwargs["model"] is fake_model
    assert kwargs["top_n"] == 3
    assert kwargs["file_path"].suffix == ".wav"
