from __future__ import annotations

import argparse
from pathlib import Path

from inference import load_genre_model
from preprocessing import PreprocessConfig, SpectrogramConfig

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="GenreClassifier",
        description="Predict the genre of an audio file.",
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display.",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preprocess_config = PreprocessConfig()
    spectrogram_config = SpectrogramConfig()

    try:
        model = load_genre_model()

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc