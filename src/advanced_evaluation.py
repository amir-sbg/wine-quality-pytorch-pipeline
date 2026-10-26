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


def conformal_prediction_sets(
    calibration_labels: np.ndarray,
    calibration_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    alpha: float = 0.1,
    test_labels: np.ndarray | None = None,
) -> dict[str, float | int | list[int]]:
    """Build finite-sample binary prediction sets from validation nonconformity."""

    calibration_labels, calibration_probabilities = _binary_arrays(
        calibration_labels,
        calibration_probabilities,
    )
    test_probabilities = np.asarray(test_probabilities, dtype=np.float64)
    if test_probabilities.ndim != 1 or len(test_probabilities) == 0:
        raise ValueError("test_probabilities must be a non-empty vector")
    if not np.all(np.isfinite(test_probabilities)) or np.any(
        (test_probabilities < 0) | (test_probabilities > 1)
    ):
        raise ValueError("test_probabilities must be finite and between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    calibration_true_probability = np.where(
        calibration_labels == 1,
        calibration_probabilities,
        1.0 - calibration_probabilities,
    )
    scores = 1.0 - calibration_true_probability
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    threshold = float(np.quantile(scores, quantile_level, method="higher"))
    include_zero = (1.0 - test_probabilities) >= 1.0 - threshold
    include_one = test_probabilities >= 1.0 - threshold
    empty = ~(include_zero | include_one)
    include_zero = include_zero | (empty & (test_probabilities < 0.5))
    include_one = include_one | (empty & (test_probabilities >= 0.5))
    set_sizes = include_zero.astype(int) + include_one.astype(int)

    report: dict[str, float | int | list[int]] = {
        "alpha": float(alpha),
        "nonconformity_threshold": threshold,
        "average_set_size": float(set_sizes.mean()),
        "argmax_fallbacks": int(empty.sum()),
        "empty_sets": int(np.sum(set_sizes == 0)),
        "singleton_sets": int(np.sum(set_sizes == 1)),
        "doubleton_sets": int(np.sum(set_sizes == 2)),
        "set_sizes": set_sizes.tolist(),
    }
    if test_labels is not None:
        test_labels = np.asarray(test_labels, dtype=int)
        if test_labels.shape != test_probabilities.shape or not set(np.unique(test_labels)).issubset({0, 1}):
            raise ValueError("test_labels must be a binary vector aligned with test probabilities")
        covered = np.where(test_labels == 1, include_one, include_zero)
        report["empirical_coverage"] = float(covered.mean())
    return report


def mahalanobis_shift_report(
    train_features: np.ndarray,
    test_features: np.ndarray,
    quantile: float = 0.95,
    shrinkage: float = 0.05,
    top_k: int = 10,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    """Flag test rows far from the multivariate training distribution."""

    train_features = np.asarray(train_features, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("train_features and test_features must be matrices")
    if train_features.shape[1] != test_features.shape[1] or len(train_features) < 2:
        raise ValueError("train and test features must share columns and include training rows")
    if not np.all(np.isfinite(train_features)) or not np.all(np.isfinite(test_features)):
        raise ValueError("features must contain only finite values")
    if not 0.0 < quantile < 1.0 or not 0.0 <= shrinkage <= 1.0 or top_k < 1:
        raise ValueError("quantile, shrinkage, and top_k are outside their valid ranges")

    mean = train_features.mean(axis=0)
    covariance = np.cov(train_features, rowvar=False)
    covariance = np.atleast_2d(covariance)
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(regularized)

    def distances(values: np.ndarray) -> np.ndarray:
        centered = values - mean
        return np.einsum("ij,jk,ik->i", centered, precision, centered)

    train_scores = distances(train_features)
    test_scores = distances(test_features)
    threshold = float(np.quantile(train_scores, quantile))
    order = np.argsort(-test_scores, kind="stable")[: min(top_k, len(test_scores))]
    return {
        "threshold_quantile": float(quantile),
        "distance_threshold": threshold,
        "train_mean_distance": float(train_scores.mean()),
        "test_mean_distance": float(test_scores.mean()),
        "flagged_test_rows": int(np.sum(test_scores > threshold)),
        "flagged_test_rate": float(np.mean(test_scores > threshold)),
        "top_test_rows": [
            {"index": int(index), "squared_mahalanobis_distance": float(test_scores[index])}
            for index in order
        ],
    }


def fgsm_robustness_report(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    epsilons: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1),
    threshold: float = 0.5,
) -> dict[str, int | list[dict[str, float]]]:
    """Stress standardized inputs in the direction that increases BCE loss."""

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features and labels must be aligned matrix and vector inputs")
    if not epsilons or any(value < 0.0 for value in epsilons):
        raise ValueError("epsilons must contain non-negative values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    was_training = model.training
    model.eval()
    x = torch.from_numpy(features).to(device).requires_grad_(True)
    y = torch.from_numpy(labels).to(device)
    logits = model(x).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    direction = x.grad.detach().sign()
    with torch.no_grad():
        baseline = torch.sigmoid(logits.detach()).cpu().numpy()
        baseline_predictions = baseline >= threshold
        rows = []
        for epsilon in epsilons:
            perturbed_logits = model(x.detach() + epsilon * direction).squeeze(-1)
            probabilities = torch.sigmoid(perturbed_logits).cpu().numpy()
            predictions = probabilities >= threshold
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "accuracy": float(np.mean(predictions == labels)),
                    "prediction_flip_rate": float(np.mean(predictions != baseline_predictions)),
                    "mean_probability_shift": float(np.mean(np.abs(probabilities - baseline))),
                }
            )
    model.train(was_training)
    return {"n_samples": int(len(features)), "epsilons": rows}


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return error


def bootstrap_calibration_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
    samples: int = 500,
    seed: int = 42,
) -> dict[str, float | int]:
    """Attach bootstrap uncertainty to expected calibration error."""

    labels, probabilities = _binary_arrays(labels, probabilities)
    if bins < 2 or samples < 2:
        raise ValueError("bins and samples must both be at least 2")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        values.append(
            _expected_calibration_error(labels[indices], probabilities[indices], bins)
        )
    values = np.asarray(values)
    return {
        "bins": int(bins),
        "bootstrap_samples": int(samples),
        "expected_calibration_error": _expected_calibration_error(labels, probabilities, bins),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def group_robustness_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
    min_group_size: int = 5,
) -> dict[str, float | int | str | list[dict[str, float | int | str]]]:
    """Report worst-group classification quality for a supplied partition."""

    labels, probabilities = _binary_arrays(labels, probabilities)
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("groups must be a vector aligned with labels")
    if min_group_size < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_group_size and threshold are outside their valid ranges")

    predictions = (probabilities >= threshold).astype(int)
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        if int(mask.sum()) < min_group_size:
            continue
        true_positive = int(np.sum((predictions[mask] == 1) & (labels[mask] == 1)))
        false_positive = int(np.sum((predictions[mask] == 1) & (labels[mask] == 0)))
        false_negative = int(np.sum((predictions[mask] == 0) & (labels[mask] == 1)))
        denominator = 2 * true_positive + false_positive + false_negative
        rows.append(
            {
                "group": str(group),
                "count": int(mask.sum()),
                "positive_rate": float(labels[mask].mean()),
                "accuracy": float(np.mean(predictions[mask] == labels[mask])),
                "f1": float(2 * true_positive / denominator if denominator else 0.0),
            }
        )
    if not rows:
        raise ValueError("no groups meet min_group_size")
    worst = min(rows, key=lambda row: row["accuracy"])
    accuracies = [float(row["accuracy"]) for row in rows]
    return {
        "groups_evaluated": len(rows),
        "worst_group": str(worst["group"]),
        "worst_group_accuracy": float(worst["accuracy"]),
        "accuracy_gap": float(max(accuracies) - min(accuracies)),
        "groups": rows,
    }


def class_conditional_error_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | dict[str, float | int]]:
    """Separate confidence and error behavior for low- and high-quality wines."""

    labels, probabilities = _binary_arrays(labels, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    predictions = (probabilities >= threshold).astype(int)
    rows = {}
    for class_value, name in ((0, "negative"), (1, "positive")):
        mask = labels == class_value
        if not np.any(mask):
            rows[name] = {"count": 0, "error_rate": 0.0, "mean_true_class_probability": 0.0}
            continue
        true_class_probability = (
            probabilities[mask] if class_value == 1 else 1.0 - probabilities[mask]
        )
        rows[name] = {
            "count": int(mask.sum()),
            "error_rate": float(np.mean(predictions[mask] != class_value)),
            "mean_true_class_probability": float(true_class_probability.mean()),
            "p10_true_class_probability": float(np.quantile(true_class_probability, 0.1)),
            "brier_score": float(np.mean((probabilities[mask] - labels[mask]) ** 2)),
        }
    return {
        "threshold": float(threshold),
        "error_rate_gap": float(abs(rows["positive"]["error_rate"] - rows["negative"]["error_rate"])),
        "classes": rows,
    }
