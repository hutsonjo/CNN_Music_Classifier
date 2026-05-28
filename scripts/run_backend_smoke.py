"""Run an end-to-end smoke test through the Flask backend."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_AUDIO = PROJECT_ROOT / "sample_data" / "sample.mp3"


def main() -> int:
    """Upload sample audio through Flask and verify predictions."""

    # Lazy import so script is importable without the package installed.
    try:
        from music_classifier.web import create_app
    except ImportError as exc:
        print(
            f"[ERROR] Cannot import music_classifier: {exc}\n"
            "Make sure the package is installed:  pip install -e .[dev]",
            file=sys.stderr,
        )
        return 1

    if not SAMPLE_AUDIO.exists():
        raise FileNotFoundError(f"Sample audio not found: {SAMPLE_AUDIO}")

    print("Creating Flask app and loading model...")
    app = create_app()
    client = app.test_client()

    print(f"Uploading sample audio to /predict: {SAMPLE_AUDIO}")

    with SAMPLE_AUDIO.open("rb") as audio_file:
        response = client.post(
            "/predict",
            data={
                "file": (audio_file, SAMPLE_AUDIO.name),
                "top_n": "3",
            },
            content_type="multipart/form-data",
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Expected status 200, got {response.status_code}: "
            f"{response.get_data(as_text=True)}"
        )

    data = response.get_json()

    if not data or "predictions" not in data:
        raise RuntimeError(f"Invalid response payload: {data}")

    predictions = data["predictions"]

    if not predictions:
        raise RuntimeError("No predictions returned.")

    print("\nFlask smoke test passed.")
    print("\nPredictions:")

    for prediction in predictions:
        genre = prediction["genre"]
        confidence = prediction["confidence"]
        print(f"{genre:<12} {confidence:.3f}")

    return 0


if __name__ == "__main__":
    main()