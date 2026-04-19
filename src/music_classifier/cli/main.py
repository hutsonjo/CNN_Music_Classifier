""""""

from __future__ import annotations

import argparse
from pathlib import Path

from music_classifier.inference import load_genre_model, classify_file

def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="GenreClassifier",
        description="Predict the genre of an audio file.",
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file."
    )
    return parser.parse_args()

def main() -> None:
    args = build_parser()
    audio_file = args.audio_file.resolve()

    try:
        model = load_genre_model()
        results = classify_file(model, audio_file)
        print(results)

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc