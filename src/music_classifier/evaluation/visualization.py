"""Plotting utilities for genre classification evaluation.

All plots follow ML reporting best practices: labeled axes, legends,
consistent styling, and 300 DPI output for reports and presentations.

All functions accept an optional ``save_path`` argument. When provided,
the figure is saved at 300 DPI and the figure is closed automatically
(useful for batch evaluation runs that generate many plots).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PathLike = Union[str, Path, None]

# Consistent style defaults
_STYLE_DEFAULTS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}
plt.rcParams.update(_STYLE_DEFAULTS)

_GENRE_CMAP = "tab10"  # distinct colors for 10 GTZAN genres


def _save_or_show(fig, save_path: PathLike) -> None:
    """Save figure at 300 DPI or display it interactively."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


# 1. Training history curves (loss + accuracy over epochs)
def plot_training_history(history, save_path: PathLike = None) -> None:
    """Plot training and validation loss and accuracy curves."""

    h = history.history if hasattr(history, "history") else history
    epochs = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, h["loss"], "o-", label="Train Loss", markersize=3)
    ax1.plot(epochs, h["val_loss"], "o-", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, h["accuracy"], "o-", label="Train Acc", markersize=3)
    ax2.plot(epochs, h["val_accuracy"], "o-", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    fig.suptitle("Training History", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# 2. Confusion matrix heatmap
def plot_confusion_matrix(
    cm: np.ndarray,
    genre_labels: list[str],
    normalize: bool = True,
    save_path: PathLike = None,
) -> None:
    """Plot a confusion matrix heatmap showing which genres the model confuses."""
    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".1%"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix (Counts)"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=genre_labels,
        yticklabels=genre_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("True Genre")
    ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# 3. Genre distribution bar chart (dataset exploration)
def plot_genre_distribution(
    genre_labels: list[str],
    counts,
    save_path: PathLike = None,
) -> None:
    """Bar chart of clip or segment counts per genre, useful for verifying dataset balance."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap(_GENRE_CMAP)(np.linspace(0, 1, len(genre_labels)))

    bars = ax.bar(genre_labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.set_title("Genre Distribution in Dataset", fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    _save_or_show(fig, save_path)


# 4. Top-k prediction display (inference output)
def plot_top_k_predictions(
    genre_labels: list[str],
    probabilities: np.ndarray,
    k: int = 5,
    save_path: PathLike = None,
) -> None:
    """
    Horizontal bar chart of the top-k genre predictions with confidence
    scores for one sample.
    """
    top_idx = np.argsort(probabilities)[-k:][::-1]
    top_genres = [genre_labels[i] for i in top_idx]
    top_probs = probabilities[top_idx]

    fig, ax = plt.subplots(figsize=(8, 0.6 * k + 1.5))
    colors = plt.cm.Blues(np.linspace(0.85, 0.4, k))

    ax.barh(range(k), top_probs, color=colors, edgecolor="white")
    ax.set_yticks(range(k))
    ax.set_yticklabels(top_genres)
    ax.set_xlabel("Confidence")
    ax.set_title("Top Genre Predictions", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    for i, prob in enumerate(top_probs):
        ax.text(prob + 0.01, i, f"{prob:.1%}", va="center", fontsize=10)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# 5. Per-class accuracy bar chart
def plot_per_class_accuracy(
    per_class_dict: dict[str, float],
    save_path: PathLike = None,
) -> None:
    """Bar chart of genre accuracy, green/red by 70% threshold w/ mean line."""

    genres = list(per_class_dict.keys())
    accs = list(per_class_dict.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if a >= 0.7 else "#e74c3c" for a in accs]

    ax.bar(genres, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Genre Classification Accuracy", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(
        y=float(np.mean(accs)),
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {np.mean(accs):.2%}",
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.02, f"{acc:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, save_path)
