from __future__ import annotations

import numpy as np
import pytest

from music_classifier.inference import GENRE_LABELS
from music_classifier.inference.predict import format_prediction, predict_batch


# ---------------------------------------------------------------------------
# format_prediction
# ---------------------------------------------------------------------------


def test_format_prediction_aggregates_and_sorts_descending() -> None:
    predictions = np.array(
        [
            [0.10, 0.60, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.20, 0.40, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    results = format_prediction(predictions)

    assert len(results) == len(GENRE_LABELS)
    assert results[0] == ("classical", 0.5)
    assert results[1] == ("country", 0.35)
    assert results[2] == ("blues", 0.15)


def test_format_prediction_rounds_to_three_decimals() -> None:
    predictions = np.array(
        [
            [0.1234, 0.8766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1234, 0.8766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    results = format_prediction(predictions)

    assert results[0] == ("classical", 0.877)
    assert results[1] == ("blues", 0.123)


def test_format_prediction_empty_input_raises_value_error() -> None:
    predictions = np.empty((0, len(GENRE_LABELS)), dtype=np.float32)

    with pytest.raises(ValueError, match="No predictions were provided"):
        format_prediction(predictions)


def test_format_prediction_non_2d_input_raises_value_error() -> None:
    predictions = np.array([], dtype=np.float32)

    with pytest.raises(ValueError, match="Predictions must be a 2-D array"):
        format_prediction(predictions)


def test_format_prediction_wrong_class_count_raises_value_error() -> None:
    predictions = np.array([[0.5, 0.5]], dtype=np.float32)

    with pytest.raises(ValueError, match="Model output shape does not match"):
        format_prediction(predictions)


def test_format_prediction_out_of_range_confidence_raises_value_error() -> None:
    predictions = np.array(
        [[1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="Prediction confidences"):
        format_prediction(predictions)


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------


def test_predict_batch_invalid_shape_raises_value_error(mocker) -> None:
    model = mocker.Mock()
    spectrograms = np.random.rand(128, 130).astype(np.float32)

    with pytest.raises(ValueError, match="Expected spectrograms with shape"):
        predict_batch(model, spectrograms)


def test_predict_batch_adds_channel_dimension_and_calls_model_predict(mocker) -> None:
    model = mocker.Mock()
    spectrograms = np.random.rand(5, 128, 130).astype(np.float32)

    raw_predictions = np.array(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.80, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.70, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.60, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    model.predict.return_value = raw_predictions

    predict_batch(model, spectrograms)

    model.predict.assert_called_once()
    called_batch = model.predict.call_args.args[0]

    assert called_batch.shape == (5, 128, 130, 1)
    assert model.predict.call_args.kwargs["verbose"] == 0


def test_predict_batch_passes_predictions_to_format_prediction(mocker) -> None:
    model = mocker.Mock()
    spectrograms = np.random.rand(5, 128, 130).astype(np.float32)

    raw_predictions = np.array(
        [
            [0.10, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.20, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.30, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    model.predict.return_value = raw_predictions

    formatted = [("classical", 0.84), ("blues", 0.16)]
    format_mock = mocker.patch(
        "music_classifier.inference.predict.format_prediction",
        return_value=formatted,
    )

    result = predict_batch(model, spectrograms)

    format_mock.assert_called_once()
    assert result == formatted