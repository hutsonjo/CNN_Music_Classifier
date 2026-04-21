#!/usr/bin/env python3
"""Evaluate a trained CNN genre classifier against a test set.

Consumes a Keras model and a ``.npz`` dataset produced by the
``music_classifier.preprocessing`` pipeline, computes the full metric
suite, writes plots and a JSON experiment log.

Usage
-----
python scripts/run_evaluate.py \\
    --model-path src/music_classifier/model/models/V1.keras \\
    --test-npz data/processed/test.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Spectrogram shape conventions from the preprocessing tests
N_MELS = 128
N_FRAMES = 130
EXPECTED_CLIP_SHAPE = (N_MELS, N_FRAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN genre classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved Keras model (directory or .keras file).",
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--test-npz",
        type=Path,
        help="Path to a pre-saved test set .npz (from save_dataset).",
    )
    data_group.add_argument(
        "--dataset-npz",
        type=Path,
        help="Path to a full dataset .npz — a test split will be extracted.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for stratified_split when using --dataset-npz. "
             "Match the training seed for a consistent test set.",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Path to a Keras history.history JSON for training curves.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Where to write figures and the experiment log.",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help="Short tag for this run (e.g. 'V1'). Defaults to the model "
             "path's basename.",
    )
    return parser.parse_args()


def _ensure_channel_dim(X):
    """Expand (n, 128, 130) → (n, 128, 130, 1) if needed."""
    import numpy as np

    if X.ndim == 3 and X.shape[1:] == EXPECTED_CLIP_SHAPE:
        X = np.expand_dims(X, axis=-1)
        print(f"Expanded input shape to: {X.shape}")
    elif X.ndim == 4 and X.shape[1:] == EXPECTED_CLIP_SHAPE + (1,):
        pass
    else:
        raise ValueError(
            f"Unexpected input shape {X.shape}. "
            f"Expected (n, {N_MELS}, {N_FRAMES}) or (n, {N_MELS}, {N_FRAMES}, 1)."
        )
    return X


def _verify_normalization(X) -> None:
    """Warn if the input is not in [0, 1] — the team pipeline produces [0, 1]."""
    x_min, x_max = float(X.min()), float(X.max())
    if x_min < -0.01 or x_max > 1.01:
        print(
            f"[WARNING] Input range is [{x_min:.3f}, {x_max:.3f}] — "
            f"the team preprocessing pipeline produces values in [0, 1]. "
            f"Double-check preprocessing before trusting results."
        )
    else:
        print(f"Input range OK: [{x_min:.3f}, {x_max:.3f}]")


def _split_dataset_for_eval(dataset_path: Path, seed: int):
    """Extract a test split from a full dataset .npz using stratified_split.

    load_dataset returns flat arrays without file-level grouping, so this
    treats each segment as its own record when calling stratified_split.
    For true file-level splitting, save a dedicated test .npz instead.
    """
    import numpy as np

    from src.music_classifier.preprocessing.splitter import stratified_split
    from src.music_classifier.preprocessing.storage import load_dataset

    print(f"Loading full dataset: {dataset_path}")
    X, y, label_names = load_dataset(dataset_path)
    print(f"  X={X.shape}, y={y.shape}, labels={label_names}")

    # Build one pseudo-record per segment so stratified_split can handle it.
    records = [
        {
            "path": f"seg_{i:06d}",
            "label": label_names[int(y[i])],
            "spectrograms": X[i: i + 1],
            "sr": 22050,
        }
        for i in range(len(X))
    ]

    _, _, test_records = stratified_split(
        records,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
    )

    X_test = np.concatenate([r["spectrograms"] for r in test_records], axis=0)
    y_test = np.array(
        [label_names.index(r["label"]) for r in test_records],
        dtype=np.int64,
    )
    print(f"Test split: X_test={X_test.shape}, y_test={y_test.shape}")
    return X_test, y_test, label_names


def main() -> int:
    args = parse_args()

    # Lazy imports so --help works without the full ML stack installed.
    try:
        import tensorflow as tf

        from src.music_classifier.evaluation import (
            ExperimentLogger,
            evaluate_predictions,
            plot_confusion_matrix,
            plot_per_class_accuracy,
            plot_top_k_predictions,
            plot_training_history,
        )
        from src.music_classifier.preprocessing.storage import load_dataset
    except ImportError as exc:
        print(
            f"[ERROR] Cannot import required packages: {exc}\n"
            "Make sure the project is installed and TensorFlow is available:\n"
            "  pip install -e .[dev]\n"
            "  pip install tensorflow scikit-learn matplotlib seaborn",
            file=sys.stderr,
        )
        return 1


    # Resolve paths and tags
    model_path = args.model_path.resolve()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1

    version_tag = args.version_tag or model_path.name
    figures_dir = args.output_dir / "figures"
    experiments_dir = args.output_dir / "experiments"
    figures_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load test data
    if args.test_npz:
        if not args.test_npz.exists():
            print(f"[ERROR] Test .npz not found: {args.test_npz}", file=sys.stderr)
            return 1
        print(f"Loading test set: {args.test_npz}")
        X_test, y_test, label_names = load_dataset(args.test_npz)
        print(f"  X={X_test.shape}, y={y_test.shape}, labels={label_names}")
        data_source = str(args.test_npz)
    else:
        if not args.dataset_npz.exists():
            print(f"[ERROR] Dataset .npz not found: {args.dataset_npz}", file=sys.stderr)
            return 1
        X_test, y_test, label_names = _split_dataset_for_eval(
            args.dataset_npz, args.split_seed
        )
        data_source = f"{args.dataset_npz} (split seed={args.split_seed})"

    X_test = _ensure_channel_dim(X_test)
    _verify_normalization(X_test)

    # Experiment log
    logger = ExperimentLogger(experiments_dir)
    run_id = logger.start_run(
        config={
            "version_tag": version_tag,
            "model_path": str(model_path),
            "data_source": data_source,
            "n_test_segments": int(len(X_test)),
            "n_classes": len(label_names),
            "label_names": list(label_names),
        }
    )

    # Predict and evaluate
    print("\nGenerating predictions...")
    y_probs = model.predict(X_test, verbose=1)
    print(f"Prediction shape: {y_probs.shape}")

    results = evaluate_predictions(y_test, y_probs, list(label_names))
    logger.log_results(run_id, results)

    # Figures
    print("\nGenerating figures...")
    stem = f"{run_id}_{version_tag}"

    plot_confusion_matrix(
        results["confusion_matrix"],
        list(label_names),
        normalize=True,
        save_path=figures_dir / f"{stem}_confusion_matrix.png",
    )
    plot_per_class_accuracy(
        results["per_class_accuracy"],
        save_path=figures_dir / f"{stem}_per_class_accuracy.png",
    )
    plot_top_k_predictions(
        list(label_names),
        y_probs[0],
        k=5,
        save_path=figures_dir / f"{stem}_sample_top5.png",
    )

    if args.history_json:
        if args.history_json.exists():
            with open(args.history_json, "r") as f:
                history = json.load(f)
            plot_training_history(
                history,
                save_path=figures_dir / f"{stem}_training_history.png",
            )
        else:
            print(
                f"[WARNING] --history-json {args.history_json} not found — "
                f"skipping training curve plot."
            )

    # Summary across all logged runs
    print("\n" + "=" * 60)
    print("ALL LOGGED RUNS")
    print("=" * 60)
    logger.summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())