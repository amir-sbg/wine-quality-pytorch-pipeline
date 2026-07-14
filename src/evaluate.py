from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(features).float().to(device)
        return torch.sigmoid(model(tensor)).cpu().numpy()


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, float | int | None] = {
        "n_samples": int(len(labels)),
        "threshold": float(threshold),
        "positive_rate": float(predictions.mean()),
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _save_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_evaluation_outputs(
    model: nn.Module,
    test_features: np.ndarray,
    test_frame: pd.DataFrame,
    history: list[dict[str, float]],
    output_dir: Path,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = test_frame["good_quality"].to_numpy(dtype=int)
    probabilities = predict_probabilities(model, test_features, device)
    predictions = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(labels, probabilities, threshold)
    _save_json(metrics, output_dir / "metrics.json")

    predictions_frame = test_frame.reset_index(drop=True).copy()
    predictions_frame["prediction_probability"] = probabilities
    predictions_frame["prediction"] = predictions
    predictions_frame["correct"] = (
        predictions_frame["prediction"] == predictions_frame["good_quality"]
    )
    predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    _plot_learning_curve(history, output_dir / "learning_curve.png")
    _plot_confusion_matrix(labels, predictions, output_dir / "confusion_matrix.png")
    if len(np.unique(labels)) == 2:
        _plot_roc_curve(labels, probabilities, output_dir / "roc_curve.png")
    return metrics


def _plot_learning_curve(history: list[dict[str, float]], path: Path) -> None:
    history_frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    axis.plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        label="validation",
    )
    axis.set_title("Training history")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCE loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, predictions),
        display_labels=["not good", "good"],
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Test confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_roc_curve(labels: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    axis.set_title("Test ROC curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
