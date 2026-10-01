import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    FEATURE_COLUMNS,
    clean_and_label,
    feature_shift_report,
    split_dataset,
    validate_schema,
)
from src.evaluate import (
    brier_decomposition,
    bootstrap_metric_intervals,
    calibration_table,
    classification_metrics,
    feature_slice_metrics,
    operating_point_report,
    permutation_importance,
    selective_risk_report,
    threshold_search,
)
from src.model import TabularMLP
from src.pipeline import build_parser, choose_device, validate_runtime_args, write_model_card
from src.train import TrainConfig, make_loader, train_model, training_diagnostics


def sample_frame(rows: int = 12) -> pd.DataFrame:
    values = np.ones((rows, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame["quality"] = [5, 6] * (rows // 2)
    return frame


def test_schema_and_target_creation() -> None:
    frame = sample_frame()
    validate_schema(frame)
    cleaned = clean_and_label(frame, quality_threshold=6)
    assert "good_quality" in cleaned
    assert set(cleaned["good_quality"]) == {0, 1}


def test_cleaning_removes_duplicate_rows() -> None:
    frame = pd.concat([sample_frame(2), sample_frame(2).iloc[[0]]], ignore_index=True)
    cleaned = clean_and_label(frame)
    assert len(cleaned) == 2


def test_schema_failure_is_explicit() -> None:
    frame = sample_frame().drop(columns=["alcohol"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_schema(frame)


def test_split_dataset_is_stratified_and_reproducible() -> None:
    frame = sample_frame(40)
    frame.loc[:, FEATURE_COLUMNS[0]] = np.arange(len(frame))
    cleaned = clean_and_label(frame)

    first_split = split_dataset(cleaned, seed=19)
    second_split = split_dataset(cleaned, seed=19)

    assert [len(partition) for partition in first_split] == [24, 8, 8]
    for first, second in zip(first_split, second_split):
        pd.testing.assert_frame_equal(first, second)
        assert first["good_quality"].mean() == 0.5


def test_feature_shift_report_tracks_split_differences() -> None:
    frame = sample_frame(12)
    frame.loc[:, "fixed acidity"] = np.linspace(7.0, 9.0, len(frame))
    frame.loc[:, "alcohol"] = np.linspace(8.0, 13.0, len(frame))
    reference = clean_and_label(frame)
    comparison = reference.copy()
    comparison["alcohol"] = comparison["alcohol"] + 2.0

    report = feature_shift_report(
        reference,
        comparison,
        feature_columns=["alcohol", "sulphates"],
    )

    assert report["reference_rows"] == 12
    assert report["comparison_rows"] == 12
    assert report["target_rate_delta"] == 0.0
    assert report["max_abs_standardized_mean_difference"] > 0
    assert {row["feature"] for row in report["features"]} == {
        "alcohol",
        "sulphates",
    }


def test_feature_shift_report_rejects_missing_columns() -> None:
    reference = clean_and_label(sample_frame(4))

    with pytest.raises(ValueError, match="missing"):
        feature_shift_report(
            reference,
            reference.drop(columns=["alcohol"]),
            feature_columns=["alcohol"],
        )


def test_train_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=0)
    with pytest.raises(ValueError, match="gradient_clip"):
        TrainConfig(gradient_clip=0)


def test_make_loader_uses_seed_for_shuffle_order() -> None:
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.arange(12, dtype=np.float32)

    first_loader = make_loader(features, labels, batch_size=4, shuffle=True, seed=123)
    second_loader = make_loader(features, labels, batch_size=4, shuffle=True, seed=123)
    first_batch = next(iter(first_loader))[1]
    second_batch = next(iter(second_loader))[1]

    torch.testing.assert_close(first_batch, second_batch)
    assert not torch.equal(first_batch, torch.from_numpy(labels[:4]))


def test_training_diagnostics_report_loss_gaps() -> None:
    diagnostics = training_diagnostics(
        [
            {"epoch": 1.0, "train_loss": 0.70, "validation_loss": 0.76},
            {"epoch": 2.0, "train_loss": 0.58, "validation_loss": 0.62},
            {"epoch": 3.0, "train_loss": 0.50, "validation_loss": 0.66},
        ]
    )

    assert diagnostics["best_train_loss"] == 0.58
    assert diagnostics["best_generalization_gap"] == pytest.approx(0.04)
    assert diagnostics["final_generalization_gap"] == pytest.approx(0.16)
    assert diagnostics["validation_loss_reduction"] == pytest.approx(0.14)


def test_runtime_args_reject_invalid_threshold() -> None:
    args = build_parser().parse_args(["--threshold", "1.2"])

    with pytest.raises(ValueError, match="threshold"):
        validate_runtime_args(args)


def test_runtime_args_reject_negative_permutation_repeats() -> None:
    args = build_parser().parse_args(["--permutation-importance-repeats", "-1"])

    with pytest.raises(ValueError, match="permutation_importance_repeats"):
        validate_runtime_args(args)


def test_runtime_args_reject_negative_bootstrap_samples() -> None:
    args = build_parser().parse_args(["--bootstrap-samples", "-2"])

    with pytest.raises(ValueError, match="bootstrap_samples"):
        validate_runtime_args(args)


def test_runtime_args_reject_unknown_slice_features() -> None:
    args = build_parser().parse_args(["--slice-features", "alcohol", "made up"])

    with pytest.raises(ValueError, match="unknown slice features"):
        validate_runtime_args(args)


def test_write_model_card_summarizes_run(tmp_path) -> None:
    summary = {
        "device": "cpu",
        "split_sizes": {"train": 100, "validation": 30, "test": 30},
        "quality_threshold": 6,
        "selected_threshold": 0.45,
        "training": {"best_epoch": 8, "epochs_trained": 12},
        "test_metrics": {
            "accuracy": 0.80,
            "f1": 0.75,
            "balanced_accuracy": 0.78,
            "roc_auc": 0.84,
            "average_precision": 0.82,
            "brier_score": 0.16,
        },
        "permutation_importance": {
            "features": [{"feature": "alcohol", "importance": 0.12}]
        },
    }

    path = write_model_card(summary, tmp_path / "model_card.md")

    text = path.read_text(encoding="utf-8")
    assert "Wine Quality Classifier" in text
    assert "0.450" in text
    assert "`alcohol`" in text


def test_auto_device_returns_torch_device() -> None:
    assert isinstance(choose_device("auto"), torch.device)


def test_metrics_include_balanced_views() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["matthews_correlation"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["brier_score"] == pytest.approx(0.02125)
    assert metrics["log_loss"] < 0.17


def test_metrics_reject_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        classification_metrics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            threshold=1.5,
        )


def test_metrics_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        classification_metrics(np.array([]), np.array([]))


def test_metrics_reject_probability_range_errors() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        classification_metrics(np.array([0, 1]), np.array([-0.1, 0.8]))


def test_metrics_reject_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        classification_metrics(np.array([0, 2]), np.array([0.1, 0.8]))


def test_metrics_leave_balanced_accuracy_empty_for_single_class() -> None:
    metrics = classification_metrics(np.array([1, 1]), np.array([0.7, 0.8]))

    assert metrics["balanced_accuracy"] is None
    assert metrics["matthews_correlation"] is None
    assert metrics["roc_auc"] is None


def test_calibration_table_reports_bin_errors() -> None:
    report = calibration_table(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        bins=2,
    )

    assert report["n_samples"] == 4
    assert report["brier_score"] == pytest.approx(0.025)
    assert report["expected_calibration_error"] == pytest.approx(0.15)
    assert len(report["table"]) == 2


def test_brier_decomposition_separates_reliability_and_resolution() -> None:
    report = brier_decomposition(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        bins=2,
    )

    assert report["base_rate"] == 0.5
    assert report["reliability"] == 0.0
    assert report["resolution"] == 0.25
    assert report["uncertainty"] == 0.25
    assert report["reconstructed_brier"] == 0.0


def test_brier_decomposition_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        brier_decomposition(np.array([0, 1]), np.array([0.2, 0.8]), bins=1)


def test_selective_risk_drops_low_confidence_errors() -> None:
    report = selective_risk_report(
        np.array([0, 1, 1, 0]),
        np.array([0.10, 0.90, 0.49, 0.80]),
        coverages=(1.0, 0.5),
    )

    full, half = report["points"]
    assert full["selective_risk"] == 0.5
    assert half["retained"] == 2
    assert half["selective_risk"] == 0.0
    assert report["area_under_risk_coverage"] >= 0.0


def test_selective_risk_validates_coverages() -> None:
    with pytest.raises(ValueError, match="coverages"):
        selective_risk_report(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            coverages=(0.0,),
        )


def test_threshold_search_selects_validation_cutoff() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])
    result = threshold_search(labels, probabilities, metric="f1", steps=19)

    assert result["metric"] == "f1"
    assert 0.40 < result["best_threshold"] <= 0.60
    assert result["best_score"] == 1.0


def test_operating_point_report_tracks_confusion_counts() -> None:
    report = operating_point_report(
        labels=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.20, 0.55, 0.65, 0.90]),
        thresholds=(0.50, 0.70),
    )

    first = report["thresholds"][0]
    assert first["true_negative"] == 1
    assert first["false_positive"] == 1
    assert first["true_positive"] == 2
    assert report["best_by_f1"]["threshold"] == 0.5


def test_bootstrap_metric_intervals_report_uncertainty() -> None:
    intervals = bootstrap_metric_intervals(
        labels=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.10, 0.20, 0.80, 0.90]),
        threshold=0.5,
        samples=20,
        seed=7,
    )

    assert intervals["samples"] == 20
    assert intervals["used_samples"] == 20
    assert "f1" in intervals["metrics"]
    assert intervals["metrics"]["f1"]["lower"] <= intervals["metrics"]["f1"]["upper"]


def test_bootstrap_metric_intervals_validate_sample_count() -> None:
    with pytest.raises(ValueError, match="samples"):
        bootstrap_metric_intervals(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            samples=0,
        )


def test_feature_slice_metrics_report_feature_bands() -> None:
    frame = clean_and_label(sample_frame(12))
    frame["alcohol"] = np.linspace(8.0, 14.0, len(frame))
    probabilities = np.linspace(0.1, 0.9, len(frame))

    report = feature_slice_metrics(
        frame,
        probabilities,
        feature_names=("alcohol",),
        threshold=0.5,
        bins=3,
    )

    assert report["bins"] == 3
    assert sum(row["count"] for row in report["rows"]) == len(frame)
    assert {row["feature"] for row in report["rows"]} == {"alcohol"}


def test_feature_slice_metrics_reject_missing_feature() -> None:
    frame = clean_and_label(sample_frame(4))

    with pytest.raises(ValueError, match="missing"):
        feature_slice_metrics(
            frame,
            np.array([0.1, 0.2, 0.8, 0.9]),
            feature_names=("not a column",),
        )


def test_permutation_importance_ranks_used_feature_first() -> None:
    class FirstFeatureModel(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return 6.0 * (features[:, 0] - 0.5)

    features = np.array(
        [
            [0.0, 0.2],
            [0.1, 0.8],
            [0.9, 0.4],
            [1.0, 0.6],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1])

    report = permutation_importance(
        model=FirstFeatureModel(),
        features=features,
        labels=labels,
        feature_names=["signal", "noise"],
        device=torch.device("cpu"),
        repeats=3,
        seed=3,
    )

    assert report["features"][0]["feature"] == "signal"
    assert report["features"][0]["importance"] > report["features"][1]["importance"]


def test_model_returns_one_logit_per_row() -> None:
    model = TabularMLP(input_dim=len(FEATURE_COLUMNS))
    output = model(torch.zeros(5, len(FEATURE_COLUMNS)))
    assert output.shape == (5,)


def test_training_smoke() -> None:
    features = np.random.default_rng(7).normal(
        size=(20, len(FEATURE_COLUMNS))
    ).astype(np.float32)
    labels = np.array([0, 1] * 10, dtype=np.float32)
    loader = make_loader(features, labels, batch_size=5, shuffle=True)
    model = TabularMLP(input_dim=len(FEATURE_COLUMNS))
    trained, history, summary = train_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        positive_weight=1.0,
        config=TrainConfig(epochs=2, patience=2, batch_size=5),
        device=torch.device("cpu"),
    )
    assert trained is model
    assert len(history) == 2
    assert summary["epochs_trained"] == 2
    assert summary["best_epoch"] >= 1
    assert "final_generalization_gap" in summary
