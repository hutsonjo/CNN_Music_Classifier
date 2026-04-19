"""Command-line interface for music genre classification.

This module provides a simple CLI entry point for running inference on a
single audio file. It handles user input, delegates model loading and
prediction to the inference package, and prints formatted results.

Flow
----
1. Parse command-line arguments (audio file path).
2. Load the trained model.
3. Run inference on the provided file.
4. Display ranked genre predictions.

This module is intentionally thin: it does not implement inference logic.
All heavy processing is delegated to the ``music_classifier.inference`` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from music_classifier.inference import load_genre_model, classify_file


def valid_audio_file(path_str: str) -> Path:
    """validates the command-line arguments for the audio file.

    Returns
    -------
    path
        Path object leading to the audio file.

     Raises
    ------
    ArgumentTypeError
        If path does not exist or isn't a file.
    """
    path = Path(path_str)

    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} does not exist")

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a file")

    return path


def build_parser() -> argparse.Namespace:
    """Parse command-line arguments for the genre classifier.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the path to the audio file.
    """
    parser = argparse.ArgumentParser(
        prog="GenreClassifier",
        description="Predict the genre of an audio file.",
    )
    parser.add_argument(
        "audio_file",
        type=valid_audio_file,
        help="Path to the audio file."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution.

    Loads the trained model, runs inference on the provided audio file,
    and prints the resulting predictions.
    """
    args = build_parser()
    audio_file = args.audio_file

    try:
        model = load_genre_model()
        results = classify_file(model, audio_file)
        print("\nPredictions:")
        for label, score in results:
            print(f"{label:10} {score:.3f}")
        SystemExit(0)

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
