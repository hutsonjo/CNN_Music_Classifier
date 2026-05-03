"""Multi-class evaluation metrics for the GTZAN genre classifier.

Provides accuracy, F1, top-k accuracy, confusion matrix, ROC AUC,
and per-class accuracy via evaluate_model() or evaluate_predictions().
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 3) -> float:
    """Fraction of samples where the true label appears in the top-k predictions."""

    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(correct.mean())


def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, genre_labels: list[str]
) -> dict[str, float]:
    """Returns per-genre accuracy dict to spot which genres model confuses."""
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)
    if np.any(row_sums=0):
        missing = [genre_labels[i] for i, s in enumerate(row_sums) if s == 0]
        print(f"Warning: no test samples for genres: {missing}")
    
    diagonal = np.divide(
        cm.diagonal(),
        row_sums,
        out=np.zeros_like(row_sums, dtype=float),
        where=row_sums != 0,
    )
    diagonal = cm.diagonal() / cm.sum(axis=1)
    return dict(zip(genre_labels, diagonal))

def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    genre_labels: list[str],
    k_values: tuple[int, ...] = (3, 5),
) -> dict:
    """Compute the full metric suite from raw probability arrays.

    Use when you have softmax outputs directly instead of a model object.
    Returns accuracy, F1, top-k, ROC AUC, confusion matrix, and per-class accuracy.
    """
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_wt = f1_score(y_true, y_pred, average="weighted")

    top_k = {k: top_k_accuracy(y_true, y_probs, k=k) for k in k_values}

    # ROC AUC can fail if a class has no samples in the test set.
    try:
        roc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        roc = None

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=genre_labels, digits=4)
    pca = per_class_accuracy(y_true, y_pred, genre_labels)

    results = {
        "y_pred": y_pred,
        "y_probs": y_probs,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "top_k": top_k,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class_accuracy": pca,
    }

    _print_summary(results)
    return results


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    genre_labels: list[str],
    k_values: tuple[int, ...] = (3, 5),
) -> dict:
    """Run the full metric suite from a Keras model, handling prediction internally."""
    y_probs = model.predict(X_test)
    return evaluate_predictions(y_test, y_probs, genre_labels, k_values)


def _print_summary(results: dict) -> None:
    """Print a formatted summary of evaluation results to stdout."""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy:   {results['accuracy']:.4f}")
    print(f"F1 (macro):         {results['f1_macro']:.4f}")
    print(f"F1 (weighted):      {results['f1_weighted']:.4f}")
    for k, v in results["top_k"].items():
        print(f"Top-{k} Accuracy:     {v:.4f}")
    if results["roc_auc"] is not None:
        print(f"ROC AUC (macro):    {results['roc_auc']:.4f}")
    print("-" * 60)
    print(results["classification_report"])
    print("-" * 60)
    print("Per-class accuracy:")
    for genre, a in results["per_class_accuracy"].items():
        print(f"  {genre:<15s} {a:.4f}")
