"""Smoke tests for the evaluation framework with synthetic data.

These tests verify that the metrics, visualization, and logging modules
work end-to-end before the real trained model is available. They use
synthetic ``y_true`` and ``y_probs`` arrays shaped like the real team
pipeline output ``(n_segments, 128, 130)`` so the evaluation framework
can be validated independently of model training progress.

Run:
    pytest tests/test_evaluation_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


from src.music_classifier.evaluation import (
    ExperimentLogger,
    evaluate_predictions,
    per_class_accuracy,
    plot_confusion_matrix,
    plot_genre_distribution,
    plot_per_class_accuracy,
    plot_top_k_predictions,
    plot_training_history,
    top_k_accuracy,
)

# Match the GTZAN label set used by the preprocessing pipeline.
LABEL_NAMES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
N_CLASSES = len(LABEL_NAMES)


# Helpers

def _make_synthetic_predictions(
    n_samples: int = 200,
    accuracy_target: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate plausible ``(y_true, y_probs)`` with approximate accuracy.

    Each sample gets a softmax-like probability vector where the true
    class has the highest probability ``accuracy_target`` fraction of the
    time. This produces a realistic-looking confusion matrix without
    requiring a real model.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, N_CLASSES, size=n_samples)
    y_probs = rng.dirichlet(np.ones(N_CLASSES), size=n_samples).astype(np.float32)

    # For `accuracy_target` fraction, boost the true class so it wins argmax.
    for i in range(n_samples):
        if rng.random() < accuracy_target:
            boost_idx = y_true[i]
            y_probs[i, boost_idx] = 2.0
            y_probs[i] /= y_probs[i].sum()

    return y_true, y_probs


# Individual metric helpers

def test_top_k_accuracy_correct_bounds() -> None:
    """Top-k accuracy must be between 0 and 1, and monotonic in k."""
    y_true, y_probs = _make_synthetic_predictions(n_samples=500)

    top1 = top_k_accuracy(y_true, y_probs, k=1)
    top3 = top_k_accuracy(y_true, y_probs, k=3)
    top5 = top_k_accuracy(y_true, y_probs, k=5)

    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0
    assert 0.0 <= top5 <= 1.0
    # Higher k should never decrease accuracy.
    assert top1 <= top3 <= top5


def test_top_k_accuracy_perfect_predictions() -> None:
    """Perfect argmax predictions yield top-1 accuracy of 1.0."""
    n = 50
    y_true = np.arange(n) % N_CLASSES
    y_probs = np.zeros((n, N_CLASSES), dtype=np.float32)
    for i in range(n):
        y_probs[i, y_true[i]] = 1.0

    assert top_k_accuracy(y_true, y_probs, k=1) == 1.0


def test_per_class_accuracy_keys_match_labels() -> None:
    """Every genre label must appear in the per-class accuracy dict."""
    y_true, y_probs = _make_synthetic_predictions(n_samples=500)
    y_pred = np.argmax(y_probs, axis=1)

    pca = per_class_accuracy(y_true, y_pred, LABEL_NAMES)
    assert set(pca.keys()) == set(LABEL_NAMES)
    for acc in pca.values():
        assert 0.0 <= acc <= 1.0


# evaluate_predictions — end-to-end metric bundle

def test_evaluate_predictions_returns_expected_keys() -> None:
    """The results dict must contain every documented key."""
    y_true, y_probs = _make_synthetic_predictions(n_samples=200)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)

    expected_keys = {
        "y_pred", "y_probs", "accuracy", "f1_macro", "f1_weighted",
        "top_k", "roc_auc", "confusion_matrix", "classification_report",
        "per_class_accuracy",
    }
    assert expected_keys.issubset(results.keys())


def test_evaluate_predictions_confusion_matrix_shape() -> None:
    """Confusion matrix shape must be (n_classes, n_classes)."""
    y_true, y_probs = _make_synthetic_predictions(n_samples=200)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)
    assert results["confusion_matrix"].shape == (N_CLASSES, N_CLASSES)


def test_evaluate_predictions_f1_within_bounds() -> None:
    y_true, y_probs = _make_synthetic_predictions(n_samples=300)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)
    assert 0.0 <= results["f1_macro"] <= 1.0
    assert 0.0 <= results["f1_weighted"] <= 1.0


def test_evaluate_predictions_accuracy_in_expected_range() -> None:
    """Synthetic data targets ~70 % accuracy — allow generous tolerance."""
    y_true, y_probs = _make_synthetic_predictions(
        n_samples=1000, accuracy_target=0.7, seed=1
    )
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)
    assert 0.5 <= results["accuracy"] <= 0.9


# Visualization — produce PNGs to a tmp_path

def test_plot_confusion_matrix_writes_png(tmp_path: Path) -> None:
    y_true, y_probs = _make_synthetic_predictions(n_samples=200)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)

    out = tmp_path / "cm.png"
    plot_confusion_matrix(results["confusion_matrix"], LABEL_NAMES, save_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_per_class_accuracy_writes_png(tmp_path: Path) -> None:
    y_true, y_probs = _make_synthetic_predictions(n_samples=200)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)

    out = tmp_path / "pca.png"
    plot_per_class_accuracy(results["per_class_accuracy"], save_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_top_k_predictions_writes_png(tmp_path: Path) -> None:
    _, y_probs = _make_synthetic_predictions(n_samples=10)
    out = tmp_path / "topk.png"
    plot_top_k_predictions(LABEL_NAMES, y_probs[0], k=5, save_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_genre_distribution_writes_png(tmp_path: Path) -> None:
    counts = [100] * N_CLASSES  # GTZAN's balanced distribution
    out = tmp_path / "dist.png"
    plot_genre_distribution(LABEL_NAMES, counts, save_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_training_history_writes_png(tmp_path: Path) -> None:
    """Must accept the dict form that load-from-JSON produces."""
    history = {
        "loss": [2.1, 1.5, 1.1, 0.9, 0.8],
        "val_loss": [2.2, 1.7, 1.3, 1.1, 1.0],
        "accuracy": [0.3, 0.5, 0.65, 0.72, 0.76],
        "val_accuracy": [0.28, 0.48, 0.6, 0.68, 0.71],
    }
    out = tmp_path / "history.png"
    plot_training_history(history, save_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


# ExperimentLogger — JSON persistence and summary

def test_logger_writes_and_reloads_json(tmp_path: Path) -> None:
    y_true, y_probs = _make_synthetic_predictions(n_samples=100)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)

    logger = ExperimentLogger(tmp_path)
    run_id = logger.start_run(config={"model_version": "V1", "test": True})
    logger.log_results(run_id, results)

    log_file = tmp_path / "experiment_log.json"
    assert log_file.exists()

    with open(log_file, "r") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["run_id"] == run_id
    assert data[0]["status"] == "completed"
    assert "accuracy" in data[0]["results"]
    # Large arrays must NOT be in the log file.
    assert "y_pred" not in data[0]["results"]
    assert "y_probs" not in data[0]["results"]


def test_logger_persists_across_instances(tmp_path: Path) -> None:
    """A new logger instance must see runs written by a previous one."""
    y_true, y_probs = _make_synthetic_predictions(n_samples=50)
    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)

    logger_a = ExperimentLogger(tmp_path)
    run_id = logger_a.start_run(config={"model_version": "V1"})
    logger_a.log_results(run_id, results)

    logger_b = ExperimentLogger(tmp_path)
    assert len(logger_b.runs) == 1
    assert logger_b.runs[0]["run_id"] == run_id


def test_logger_get_best_run_picks_highest_accuracy(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path)

    # Low-accuracy run
    _, probs_low = _make_synthetic_predictions(
        n_samples=200, accuracy_target=0.2, seed=1
    )
    y_low = np.random.default_rng(1).integers(0, N_CLASSES, size=200)
    run_low = logger.start_run(config={"model_version": "V1"})
    logger.log_results(
        run_low, evaluate_predictions(y_low, probs_low, LABEL_NAMES)
    )

    # High-accuracy run
    y_high, probs_high = _make_synthetic_predictions(
        n_samples=200, accuracy_target=0.9, seed=2
    )
    run_high = logger.start_run(config={"model_version": "V2"})
    logger.log_results(
        run_high, evaluate_predictions(y_high, probs_high, LABEL_NAMES)
    )

    best = logger.get_best_run(metric="accuracy")
    assert best is not None
    assert best["run_id"] == run_high


def test_logger_get_best_run_empty_returns_none(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path)
    assert logger.get_best_run() is None


# End-to-end — metrics + all plots + log in one flow

def test_full_evaluation_flow_end_to_end(tmp_path: Path) -> None:
    """Run the full pipeline against synthetic data with no real model."""
    figures_dir = tmp_path / "figures"
    experiments_dir = tmp_path / "experiments"
    figures_dir.mkdir()
    experiments_dir.mkdir()

    y_true, y_probs = _make_synthetic_predictions(n_samples=500)

    logger = ExperimentLogger(experiments_dir)
    run_id = logger.start_run(config={
        "version_tag": "smoke",
        "n_test_segments": len(y_true),
    })

    results = evaluate_predictions(y_true, y_probs, LABEL_NAMES)
    logger.log_results(run_id, results)

    plot_confusion_matrix(
        results["confusion_matrix"], LABEL_NAMES,
        save_path=figures_dir / "cm.png",
    )
    plot_per_class_accuracy(
        results["per_class_accuracy"],
        save_path=figures_dir / "pca.png",
    )
    plot_top_k_predictions(
        LABEL_NAMES, y_probs[0], k=5,
        save_path=figures_dir / "topk.png",
    )

    # Verify all expected output files exist
    assert (figures_dir / "cm.png").exists()
    assert (figures_dir / "pca.png").exists()
    assert (figures_dir / "topk.png").exists()
    assert (experiments_dir / "experiment_log.json").exists()

    # Verify logger.summary() runs without crashing
    logger.summary()