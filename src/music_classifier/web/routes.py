"""Flask API routes for the music genre classifier backend.

This module defines HTTP endpoints for health checks and
music genre prediction using the inference engine.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from flask import Blueprint, jsonify, request, current_app
from flask.typing import ResponseReturnValue
from keras import Model

from music_classifier.inference import (
    classify_file,
    load_genre_model,
)

bp = Blueprint("api", __name__)


@bp.get("/health")
def health() -> ResponseReturnValue:
    """Simple health-check endpoint."""

    return {"status": "ok"}, 200


@bp.post("/predict")
def predict() -> ResponseReturnValue:
    """Run genre prediction on an uploaded audio file.

    This endpoint accepts a multipart form upload containing an audio
    file and an optional ``top_n`` parameter. The uploaded file is
    temporarily saved to disk, passed through the inference pipeline,
    and deleted after processing.

    Request
    -------
    POST /predict

    Form fields
    -----------
    file:
        Uploaded audio file.
    top_n:
        Optional integer limiting the number of returned predictions.

    Returns
    -------
    ResponseReturnValue
        JSON response containing ranked genre predictions and
        confidence scores.

    Raises
    ------
    ValueError
        Returned as a 400 response if invalid inference parameters
        are provided.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    filename = uploaded_file.filename

    if not filename:
        return jsonify({"error": "Empty filename"}), 400

    temp_path: Path | None = None

    try:
        top_n = request.form.get("top_n", default=None, type=int)
        suffix = Path(filename).suffix or ".tmp"

        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            uploaded_file.save(tmp.name)
            temp_path = Path(tmp.name)

        model: Model = current_app.config["MODEL"]

        predictions = classify_file(
            model=model,
            file_path=temp_path,
            top_n=top_n,
        )

        return jsonify(
            {
                "predictions": [
                    {
                        "genre": genre,
                        "confidence": float(confidence),
                    }
                    for genre, confidence in predictions
                ]
            }
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    finally:
        if temp_path is not None:
            if temp_path.exists():
                temp_path.unlink()
