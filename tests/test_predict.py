from __future__ import annotations

import numpy as np
import pytest

from music_classifier.inference import GENRE_LABELS
from music_classifier.inference.predict import format_prediction, predict_batch


# ---------------------------------------------------------------------------
# format_prediction
# ---------------------------------------------------------------------------


def test_format_prediction_averages_logits_applies_softmax_and_sorts_descending() -> None:
    predictions = np.array(
        [
            [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    results = format_prediction(predictions)

    assert len(results) == len(GENRE_LABELS)
    assert results[0][0] == "classical"
    assert results[1][0] == "country"
    assert results[2][0] == "blues"


def test_format_prediction_returns_softmax_probabilities() -> None:
    predictions = np.array(
        [
            [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    results = format_prediction(predictions)
    scores = [score for _, score in results]

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert np.isclose(sum(scores), 1.0, atol=0.01)


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