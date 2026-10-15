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
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from torch import nn


THRESHOLD_METRICS = {"f1", "balanced_accuracy", "matthews_correlation"}


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
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if len(labels) == 0:
        raise ValueError("labels and probabilities must not be empty")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be between 0 and 1")
    unique_labels = np.unique(labels)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError("labels must be binary")
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, float | int | None] = {
        "n_samples": int(len(labels)),
        "threshold": float(threshold),
        "positive_rate": float(predictions.mean()),
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "matthews_correlation": None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "balanced_accuracy": None,
        "specificity": None,
        "false_positive_rate": None,
    }

    if len(unique_labels) == 2:
        true_negative, false_positive, _, _ = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(labels, predictions)
        )
        metrics["matthews_correlation"] = float(
            matthews_corrcoef(labels, predictions)
        )
        negative_count = true_negative + false_positive
        metrics["specificity"] = float(
            true_negative / negative_count if negative_count else 0.0
        )
        metrics["false_positive_rate"] = float(
            false_positive / negative_count if negative_count else 0.0
        )
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def calibration_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    classification_metrics(labels, probabilities)

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    rows = []
    expected_calibration_error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(labels[mask].mean())
        error = observed_rate - mean_probability
        expected_calibration_error += count / len(labels) * abs(error)
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "calibration_error": error,
            }
        )
    return {
        "bins": bins,
        "n_samples": int(len(labels)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": float(expected_calibration_error),
        "table": rows,
    }


def calibration_slope_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    """Fit a logistic recalibration model; ideal intercept and slope are 0 and 1."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities)
    if len(np.unique(labels)) < 2:
        return {
            "n_samples": int(len(labels)),
            "calibration_intercept": None,
            "calibration_slope": None,
        }

    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    recalibrator = LogisticRegression(C=1e6, solver="lbfgs")
    recalibrator.fit(logits, labels)
    return {
        "n_samples": int(len(labels)),
        "calibration_intercept": float(recalibrator.intercept_[0]),
        "calibration_slope": float(recalibrator.coef_[0, 0]),
    }


def brier_decomposition(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> dict[str, float | int]:
    """Decompose the Brier score into reliability, resolution, and uncertainty."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities)

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    base_rate = float(labels.mean())
    reliability = 0.0
    resolution = 0.0
    populated_bins = 0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        populated_bins += 1
        weight = count / len(labels)
        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(labels[mask].mean())
        reliability += weight * (mean_probability - observed_rate) ** 2
        resolution += weight * (observed_rate - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    reconstructed = reliability - resolution + uncertainty
    score = float(brier_score_loss(labels, probabilities))
    return {
        "bins": bins,
        "populated_bins": populated_bins,
        "base_rate": base_rate,
        "brier_score": score,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "reconstructed_brier": float(reconstructed),
        "binning_residual": float(score - reconstructed),
    }


def selective_risk_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> dict[str, float | int | list[dict[str, float | int]]]:
    """Measure error after retaining only the most confident predictions."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities)
    if not coverages or any(not 0.0 < value <= 1.0 for value in coverages):
        raise ValueError("coverages must be in (0, 1]")

    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    ordered_errors = (predictions[order] != labels[order]).astype(float)
    prefix_risk = np.cumsum(ordered_errors) / np.arange(1, len(labels) + 1)

    rows = []
    for target in coverages:
        retained = min(len(labels), max(1, int(np.ceil(target * len(labels)))))
        selected = order[:retained]
        errors = int((predictions[selected] != labels[selected]).sum())
        risk = errors / retained
        rows.append(
            {
                "target_coverage": float(target),
                "coverage": float(retained / len(labels)),
                "retained": retained,
                "errors": errors,
                "selective_accuracy": float(1.0 - risk),
                "selective_risk": float(risk),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )

    return {
        "n_samples": int(len(labels)),
        "area_under_risk_coverage": float(prefix_risk.mean()),
        "points": rows,
    }


def threshold_search(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
    steps: int = 181,
) -> dict[str, float | str | list[dict[str, float]]]:
    if metric not in THRESHOLD_METRICS:
        raise ValueError(f"metric must be one of {sorted(THRESHOLD_METRICS)}")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    rows = []
    for threshold in np.linspace(0.05, 0.95, steps):
        score = classification_metrics(labels, probabilities, float(threshold))[metric]
        if score is None:
            raise ValueError(f"metric is unavailable for this label distribution: {metric}")
        rows.append({"threshold": float(threshold), "score": float(score)})

    best = max(rows, key=lambda row: (row["score"], -abs(row["threshold"] - 0.5)))
    return {
        "metric": metric,
        "best_threshold": best["threshold"],
        "best_score": best["score"],
        "thresholds": rows,
    }


def bootstrap_threshold_stability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
    steps: int = 181,
    samples: int = 250,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Measure how much the selected validation threshold moves under resampling."""

    if samples < 1:
        raise ValueError("samples must be at least 1")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities)
    generator = np.random.default_rng(seed)
    thresholds = []
    scores = []
    skipped = 0
    attempts = 0
    while len(thresholds) < samples and attempts < samples * 20:
        attempts += 1
        indices = generator.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[indices])) < 2:
            skipped += 1
            continue
        result = threshold_search(
            labels[indices],
            probabilities[indices],
            metric=metric,
            steps=steps,
        )
        thresholds.append(float(result["best_threshold"]))
        scores.append(float(result["best_score"]))

    if not thresholds:
        raise ValueError("could not draw a threshold bootstrap sample with both labels")
    values = np.asarray(thresholds)
    grid_low, grid_high = 0.05, 0.95
    return {
        "metric": metric,
        "samples": samples,
        "used_samples": len(thresholds),
        "skipped_degenerate_samples": skipped,
        "median_threshold": float(np.median(values)),
        "mean_threshold": float(values.mean()),
        "threshold_std": float(values.std(ddof=0)),
        "threshold_q10": float(np.quantile(values, 0.10)),
        "threshold_q90": float(np.quantile(values, 0.90)),
        "boundary_selection_rate": float(
            np.mean(np.isclose(values, grid_low) | np.isclose(values, grid_high))
        ),
        "mean_bootstrap_score": float(np.mean(scores)),
    }


def operating_point_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = (0.25, 0.35, 0.50, 0.65, 0.75),
) -> dict[str, list[dict[str, float | int | None]] | dict[str, float | int | None]]:
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    rows = []
    for threshold in thresholds:
        metrics = classification_metrics(labels, probabilities, threshold)
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        )
        rows.append(
            {
                "threshold": float(threshold),
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
                "positive_rate": metrics["positive_rate"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )

    best_by_f1 = max(
        rows,
        key=lambda row: (float(row["f1"]), -abs(float(row["threshold"]) - 0.5)),
    )
    available_balanced = [row for row in rows if row["balanced_accuracy"] is not None]
    best_by_balanced = (
        max(available_balanced, key=lambda row: float(row["balanced_accuracy"]))
        if available_balanced
        else best_by_f1
    )
    return {
        "thresholds": rows,
        "best_by_f1": best_by_f1,
        "best_by_balanced_accuracy": best_by_balanced,
    }


def decision_curve_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
) -> dict[str, float | int | list[dict[str, float]]]:
    """Compare model net benefit with treat-all and treat-none policies."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities)
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError("decision thresholds must be in (0, 1)")

    rows = []
    prevalence = float(labels.mean())
    sample_count = len(labels)
    for threshold in thresholds:
        predictions = probabilities >= threshold
        true_positive = int(np.sum(predictions & (labels == 1)))
        false_positive = int(np.sum(predictions & (labels == 0)))
        odds = threshold / (1.0 - threshold)
        model_net_benefit = true_positive / sample_count - false_positive / sample_count * odds
        treat_all_net_benefit = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": float(model_net_benefit),
                "treat_all_net_benefit": float(treat_all_net_benefit),
                "treat_none_net_benefit": 0.0,
                "net_benefit_vs_best_default": float(
                    model_net_benefit - max(treat_all_net_benefit, 0.0)
                ),
            }
        )
    best = max(rows, key=lambda row: row["net_benefit_vs_best_default"])
    return {
        "n_samples": sample_count,
        "prevalence": prevalence,
        "best_threshold": best["threshold"],
        "best_net_benefit_vs_default": best["net_benefit_vs_best_default"],
        "points": rows,
    }


def bootstrap_metric_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    samples: int = 250,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | dict[str, dict[str, float]]]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities, threshold)

    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    skipped_degenerate_samples = 0
    draws = 0
    attempts = 0
    needs_two_classes = len(np.unique(labels)) == 2
    max_attempts = samples * 20

    while draws < samples and attempts < max_attempts:
        attempts += 1
        indices = generator.integers(0, len(labels), size=len(labels))
        if needs_two_classes and len(np.unique(labels[indices])) < 2:
            skipped_degenerate_samples += 1
            continue
        metrics = classification_metrics(
            labels[indices],
            probabilities[indices],
            threshold,
        )
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                collected.setdefault(name, []).append(float(value))
        draws += 1

    if draws == 0:
        raise ValueError("could not draw a valid bootstrap sample")

    alpha = (1.0 - confidence) / 2.0
    intervals = {}
    for name, values in collected.items():
        sample_values = np.asarray(values, dtype=float)
        intervals[name] = {
            "mean": float(sample_values.mean()),
            "lower": float(np.quantile(sample_values, alpha)),
            "upper": float(np.quantile(sample_values, 1.0 - alpha)),
        }
    return {
        "samples": samples,
        "used_samples": draws,
        "skipped_degenerate_samples": skipped_degenerate_samples,
        "confidence": confidence,
        "threshold": float(threshold),
        "metrics": intervals,
    }


def feature_slice_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    feature_names: tuple[str, ...],
    threshold: float = 0.5,
    bins: int = 3,
) -> dict[str, int | float | list[dict[str, float | int | str | None]]]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if "good_quality" not in frame:
        raise ValueError("frame must include good_quality")
    missing = sorted(set(feature_names).difference(frame.columns))
    if missing:
        raise ValueError(f"slice features are missing from the frame: {missing}")

    labels = frame["good_quality"].to_numpy(dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classification_metrics(labels, probabilities, threshold)

    rows = []
    for feature in feature_names:
        values = pd.to_numeric(frame[feature], errors="raise").reset_index(drop=True)
        bucket_count = min(bins, int(values.nunique()))
        if bucket_count < 2:
            buckets = pd.Series(["all"], index=values.index)
        else:
            buckets = pd.qcut(values, q=bucket_count, duplicates="drop")

        working = pd.DataFrame(
            {
                "label": labels,
                "probability": probabilities,
                "value": values,
                "bucket": buckets.astype(str),
            }
        )
        for bucket, group in working.groupby("bucket", sort=False):
            rows.append(
                {
                    "feature": feature,
                    "bucket": str(bucket),
                    "count": int(len(group)),
                    "min_value": float(group["value"].min()),
                    "max_value": float(group["value"].max()),
                    **_binary_slice_metrics(
                        group["label"].to_numpy(dtype=int),
                        group["probability"].to_numpy(dtype=float),
                        threshold,
                    ),
                }
            )

    return {
        "threshold": float(threshold),
        "bins": bins,
        "rows": rows,
    }


def _binary_slice_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(int)
    result: dict[str, float | None] = {
        "positive_rate": float(predictions.mean()),
        "mean_probability": float(probabilities.mean()),
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy": None,
    }
    if len(np.unique(labels)) == 2:
        result["balanced_accuracy"] = float(
            balanced_accuracy_score(labels, predictions)
        )
    return result


def permutation_importance(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    threshold: float = 0.5,
    metric: str = "f1",
    repeats: int = 5,
    seed: int = 42,
) -> dict[str, float | str | list[dict[str, float | str]]]:
    if metric not in THRESHOLD_METRICS | {"roc_auc", "average_precision"}:
        raise ValueError("unsupported permutation metric")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of feature columns")

    labels = np.asarray(labels, dtype=int)
    baseline_probabilities = predict_probabilities(model, features, device)
    baseline_metrics = classification_metrics(labels, baseline_probabilities, threshold)
    baseline_score = baseline_metrics[metric]
    if baseline_score is None:
        raise ValueError(f"metric is unavailable for this label distribution: {metric}")

    generator = np.random.default_rng(seed)
    rows = []
    for column_index, feature_name in enumerate(feature_names):
        scores = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[:, column_index] = generator.permutation(shuffled[:, column_index])
            probabilities = predict_probabilities(model, shuffled, device)
            shuffled_metrics = classification_metrics(labels, probabilities, threshold)
            score = shuffled_metrics[metric]
            if score is None:
                raise ValueError(f"metric is unavailable after permutation: {metric}")
            scores.append(float(score))
        scores_array = np.asarray(scores, dtype=float)
        rows.append(
            {
                "feature": feature_name,
                "baseline_score": float(baseline_score),
                "mean_score": float(scores_array.mean()),
                "std_score": float(scores_array.std(ddof=0)),
                "importance": float(baseline_score - scores_array.mean()),
            }
        )

    rows = sorted(rows, key=lambda row: row["importance"], reverse=True)
    return {
        "metric": metric,
        "threshold": float(threshold),
        "repeats": repeats,
        "baseline_score": float(baseline_score),
        "features": rows,
    }


def _save_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_threshold_curve(
    threshold_rows: list[dict[str, float]],
    metric: str,
    path: Path,
) -> None:
    frame = pd.DataFrame(threshold_rows)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(frame["threshold"], frame["score"])
    axis.set_title("Validation threshold search")
    axis.set_xlabel("Classification threshold")
    axis.set_ylabel(metric.replace("_", " "))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_permutation_importance_plot(
    importance_rows: list[dict[str, float | str]],
    path: Path,
    top_k: int = 10,
) -> None:
    frame = pd.DataFrame(importance_rows).head(top_k).iloc[::-1]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.barh(frame["feature"], frame["importance"])
    axis.set_title("Permutation feature importance")
    axis.set_xlabel("Baseline score drop")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_evaluation_outputs(
    model: nn.Module,
    test_features: np.ndarray,
    test_frame: pd.DataFrame,
    history: list[dict[str, float]],
    output_dir: Path,
    device: torch.device,
    threshold: float = 0.5,
    bootstrap_samples: int = 250,
    seed: int = 42,
    slice_features: tuple[str, ...] = ("alcohol", "volatile acidity", "sulphates"),
    slice_bins: int = 3,
) -> dict[str, float | int | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = test_frame["good_quality"].to_numpy(dtype=int)
    probabilities = predict_probabilities(model, test_features, device)
    predictions = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(labels, probabilities, threshold)
    calibration = calibration_table(labels, probabilities)
    calibration_slope = calibration_slope_report(labels, probabilities)
    decomposition = brier_decomposition(labels, probabilities)
    selective_risk = selective_risk_report(labels, probabilities)
    operating_points = operating_point_report(labels, probabilities)
    decision_curve = decision_curve_report(labels, probabilities)
    metrics["best_operating_threshold_f1"] = operating_points["best_by_f1"]["threshold"]
    metrics["best_operating_threshold_balanced_accuracy"] = operating_points[
        "best_by_balanced_accuracy"
    ]["threshold"]
    metrics["brier_reliability"] = decomposition["reliability"]
    metrics["brier_resolution"] = decomposition["resolution"]
    metrics["area_under_risk_coverage"] = selective_risk[
        "area_under_risk_coverage"
    ]
    metrics["decision_curve_best_threshold"] = decision_curve["best_threshold"]
    metrics["decision_curve_best_net_benefit_vs_default"] = decision_curve[
        "best_net_benefit_vs_default"
    ]
    metrics["calibration_intercept"] = calibration_slope["calibration_intercept"]
    metrics["calibration_slope"] = calibration_slope["calibration_slope"]
    _save_json(metrics, output_dir / "metrics.json")
    _save_json(calibration, output_dir / "calibration.json")
    _save_json(calibration_slope, output_dir / "calibration_slope.json")
    _save_json(decomposition, output_dir / "brier_decomposition.json")
    _save_json(selective_risk, output_dir / "selective_risk.json")
    _save_json(operating_points, output_dir / "operating_points.json")
    _save_json(decision_curve, output_dir / "decision_curve.json")
    if bootstrap_samples > 0:
        intervals = bootstrap_metric_intervals(
            labels,
            probabilities,
            threshold=threshold,
            samples=bootstrap_samples,
            seed=seed,
        )
        _save_json(intervals, output_dir / "metric_intervals.json")
    if slice_features:
        slices = feature_slice_metrics(
            test_frame,
            probabilities,
            feature_names=slice_features,
            threshold=threshold,
            bins=slice_bins,
        )
        _save_json(slices, output_dir / "slice_metrics.json")

    predictions_frame = test_frame.reset_index(drop=True).copy()
    predictions_frame["prediction_probability"] = probabilities
    predictions_frame["prediction"] = predictions
    predictions_frame["correct"] = (
        predictions_frame["prediction"] == predictions_frame["good_quality"]
    )
    predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    _plot_learning_curve(history, output_dir / "learning_curve.png")
    _plot_confusion_matrix(labels, predictions, output_dir / "confusion_matrix.png")
    _plot_calibration(calibration, output_dir / "calibration_curve.png")
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


def _plot_calibration(calibration: dict, path: Path) -> None:
    table = calibration["table"]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    if table:
        frame = pd.DataFrame(table)
        axis.plot(
            frame["mean_probability"],
            frame["observed_positive_rate"],
            marker="o",
            label="model",
        )
    axis.set_title("Probability calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
