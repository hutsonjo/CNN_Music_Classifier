"""Temporary stub backend for frontend development.

  THIS IS A DEV-ONLY STUB.
  Use the real backend instead once it's wired up:
      flask --app src/music_classifier/web/app.py run

Purpose:
    Lets the React frontend be developed without a trained model on disk.
    Returns hardcoded fake predictions in the same JSON shape the real
    backend uses, so the frontend code requires zero changes when the
    swap happens.

Usage:
    pip install flask flask-cors
    python scripts/dev_backend_stub.py
    # Runs on http://localhost:5000

    # In another terminal:
    cd frontend && npm run dev

Contract (matches src/music_classifier/web/routes.py):
    POST /predict
      multipart/form-data with a "file" field (and optional "top_n")
      → 200 { "predictions": [{ "genre": str, "confidence": float }, ...] }
      → 400 { "error": str }

    GET /health
      → 200 { "status": "ok" }
"""
from __future__ import annotations

import time
from flask import Flask, jsonify, request

try:
    from flask_cors import CORS
except ImportError as exc:
    raise SystemExit(
        "flask-cors is required for the dev stub. "
        "Install with: pip install flask-cors"
    ) from exc


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Rotating fake results so each upload shows different data during dev.
# Intentionally NOT sorted by confidence so the frontend's sort step
# gets exercised.
_FAKE_RESULTS = [
    [
        {"genre": "rock",      "confidence": 0.521},
        {"genre": "metal",     "confidence": 0.231},
        {"genre": "blues",     "confidence": 0.118},
        {"genre": "country",   "confidence": 0.062},
        {"genre": "jazz",      "confidence": 0.028},
        {"genre": "pop",       "confidence": 0.020},
        {"genre": "disco",     "confidence": 0.010},
        {"genre": "hiphop",    "confidence": 0.006},
        {"genre": "reggae",    "confidence": 0.003},
        {"genre": "classical", "confidence": 0.001},
    ],
    [
        {"genre": "classical", "confidence": 0.812},
        {"genre": "jazz",      "confidence": 0.142},
        {"genre": "blues",     "confidence": 0.025},
        {"genre": "country",   "confidence": 0.010},
        {"genre": "pop",       "confidence": 0.005},
        {"genre": "rock",      "confidence": 0.003},
        {"genre": "reggae",    "confidence": 0.001},
        {"genre": "disco",     "confidence": 0.001},
        {"genre": "metal",     "confidence": 0.000},
        {"genre": "hiphop",    "confidence": 0.001},
    ],
    [
        {"genre": "hiphop",    "confidence": 0.412},
        {"genre": "pop",       "confidence": 0.281},
        {"genre": "disco",     "confidence": 0.184},
        {"genre": "reggae",    "confidence": 0.073},
        {"genre": "rock",      "confidence": 0.030},
        {"genre": "country",   "confidence": 0.012},
        {"genre": "metal",     "confidence": 0.005},
        {"genre": "blues",     "confidence": 0.002},
        {"genre": "jazz",      "confidence": 0.001},
        {"genre": "classical", "confidence": 0.000},
    ],
]
_counter = {"n": 0}


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify(error='Missing "file" field in form data'), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify(error="Empty file"), 400

    # Honor top_n the way the real backend does.
    top_n = request.form.get("top_n", default=None, type=int)

    # Simulate inference latency so the frontend loading state is visible.
    time.sleep(1.2)

    _counter["n"] += 1
    predictions = _FAKE_RESULTS[_counter["n"] % len(_FAKE_RESULTS)]
    if top_n is not None and top_n > 0:
        predictions = sorted(predictions, key=lambda p: p["confidence"], reverse=True)[:top_n]

    print(f"[stub] returning predictions[{_counter['n'] % len(_FAKE_RESULTS)}] "
          f"for upload: {uploaded.filename} (top_n={top_n})")
    return jsonify(predictions=predictions)


@app.get("/health")
def health():
    return jsonify(status="ok", note="stub backend — replace before production")


if __name__ == "__main__":
    print("=" * 60)
    print("DEV STUB BACKEND — fake predictions, no real model")
    print("Use the real backend instead once you're ready:")
    print("  flask --app src/music_classifier/web/app.py run")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)