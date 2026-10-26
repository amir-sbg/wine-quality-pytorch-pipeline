from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _binary_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    if len(labels) == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be a non-empty binary vector")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and between 0 and 1")
    return labels, probabilities


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def fit_temperature_scaling(
    labels: np.ndarray,
    probabilities: np.ndarray,
    lower: float = 0.25,
    upper: float = 5.0,
    steps: int = 300,
) -> dict[str, float]:
    """Fit one validation-set temperature by minimizing binary log loss."""

    labels, probabilities = _binary_arrays(labels, probabilities)
    if lower <= 0.0 or upper <= lower or steps < 2:
        raise ValueError("temperature search requires 0 < lower < upper and at least two steps")
    candidates = np.geomspace(lower, upper, steps)
    losses = np.array(
        [_binary_log_loss(labels, apply_temperature(probabilities, value)) for value in candidates]
    )
    best = int(np.argmin(losses))
    return {
        "temperature": float(candidates[best]),
        "log_loss_before": _binary_log_loss(labels, probabilities),
        "log_loss_after": float(losses[best]),
    }


def cost_sensitive_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    steps: int = 199,
) -> dict[str, float | int]:
    """Select a threshold from validation data for an explicit error-cost ratio."""

    labels, probabilities = _binary_arrays(labels, probabilities)
    if false_positive_cost < 0.0 or false_negative_cost < 0.0:
        raise ValueError("misclassification costs must not be negative")
    if false_positive_cost + false_negative_cost == 0.0:
        raise ValueError("at least one misclassification cost must be positive")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    best = None
    for threshold in np.linspace(0.005, 0.995, steps):
        predictions = probabilities >= threshold
        false_positives = int(np.sum((predictions == 1) & (labels == 0)))
        false_negatives = int(np.sum((predictions == 0) & (labels == 1)))
        total_cost = false_positive_cost * false_positives + false_negative_cost * false_negatives
        row = {
            "threshold": float(threshold),
            "total_cost": float(total_cost),
            "cost_per_sample": float(total_cost / len(labels)),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        if best is None or (row["total_cost"], abs(threshold - 0.5)) < (
            best["total_cost"],
            abs(best["threshold"] - 0.5),
        ):
            best = row
    return best
