"""Evaluation framework for the CNN music genre classifier.

Consumes a trained Keras model and preprocessed dataset to produce
metrics, visualizations, and a JSON experiment log.
"""

from .logger import ExperimentLogger
from .metrics import (
    evaluate_model,
    evaluate_predictions,
    per_class_accuracy,
    top_k_accuracy,
)
from .visualization import (
    plot_confusion_matrix,
    plot_genre_distribution,
    plot_per_class_accuracy,
    plot_top_k_predictions,
    plot_training_history,
)

__all__ = [
    # Metrics
    "evaluate_model",
    "evaluate_predictions",
    "per_class_accuracy",
    "top_k_accuracy",
    # Visualization
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_per_class_accuracy",
    "plot_top_k_predictions",
    "plot_genre_distribution",
    # Logging
    "ExperimentLogger",
]
