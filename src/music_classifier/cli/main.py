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
Processing is delegated to the ``music_classifier.inference`` package.
"""

from __future__ import annotations

import os

# Suppress TensorFlow/Keras logging noise before related imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

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
        Parsed command-line arguments containing:

        - ``audio_file``:
        Path to the input audio file.
        - ``top_n``:
        Optional number of top predictions to display.
    """
    parser = argparse.ArgumentParser(
        prog="GenreClassifier",
        description="Predict the genre of an audio file.",
    )
    parser.add_argument(
        "audio_file",
        type=valid_audio_file,
        help="Path to the audio file. Path may be relative to cwd or absolute."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Number of prediction results to be shown."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution.

    Loads the trained model, runs inference on the provided audio file,
    and prints the resulting top n predictions.
    """
    args = build_parser()
    audio_file = args.audio_file

    try:
        model = load_genre_model()
        results = classify_file(model, audio_file, args.top_n)

        print("\nPredictions:")
        for label, score in results:
            print(f"{label:10} {score:.3f}")
        raise SystemExit(0)

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
