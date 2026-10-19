from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .data import (
    DATA_URL,
    FEATURE_COLUMNS,
    clean_and_label,
    download_dataset,
    feature_shift_report,
    load_raw_data,
    save_json,
    split_dataset,
    summarize_dataset,
)
from .evaluate import (
    bootstrap_threshold_stability,
    permutation_importance,
    predict_probabilities,
    save_evaluation_outputs,
    save_permutation_importance_plot,
    save_threshold_curve,
    threshold_search,
)
from .model import TabularMLP
from .train import TrainConfig, make_loader, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def validate_runtime_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.permutation_importance_repeats < 0:
        raise ValueError("permutation_importance_repeats must not be negative")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must not be negative")
    if args.robustness_repeats < 0:
        raise ValueError("robustness_repeats must not be negative")
    if args.mc_dropout_passes < 0 or args.mc_dropout_passes == 1:
        raise ValueError("mc_dropout_passes must be zero or at least 2")
    if args.quality_threshold < 0:
        raise ValueError("quality_threshold must not be negative")
    missing_slice_features = sorted(set(args.slice_features).difference(FEATURE_COLUMNS))
    if missing_slice_features:
        raise ValueError(f"unknown slice features: {missing_slice_features}")
    if args.slice_bins < 2:
        raise ValueError("slice_bins must be at least 2")


def status(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[pipeline] {message}")


def save_scaler(scaler: StandardScaler, feature_names: list[str], path: Path) -> None:
    payload = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    save_json(payload, path)


def write_model_card(summary: dict, path: Path) -> Path:
    metrics = summary["test_metrics"]
    training = summary["training"]
    threshold = summary["selected_threshold"]
    top_feature = "n/a"
    importance = summary.get("permutation_importance")
    if importance and importance.get("features"):
        top_feature = importance["features"][0]["feature"]

    roc_auc = _format_metric(metrics.get("roc_auc"))
    average_precision = _format_metric(metrics.get("average_precision"))
    card = f"""# Wine Quality Classifier

Binary classifier for predicting whether a red wine is likely to score at least
{summary["quality_threshold"]} on the UCI quality scale from 11 chemistry measurements.

## Run summary

- Device: {summary["device"]}
- Split sizes: {summary["split_sizes"]["train"]} train / {summary["split_sizes"]["validation"]} validation / {summary["split_sizes"]["test"]} test
- Best epoch: {training["best_epoch"]} of {training["epochs_trained"]} trained epochs
- Decision threshold: {threshold:.3f}

## Test metrics

- Accuracy: {metrics["accuracy"]:.3f}
- F1: {metrics["f1"]:.3f}
- Balanced accuracy: {_format_metric(metrics.get("balanced_accuracy"))}
- ROC-AUC: {roc_auc}
- Average precision: {average_precision}
- Brier score: {metrics["brier_score"]:.3f}

## Notes

The preprocessing scaler is fitted only on the training split. The run also saves
calibration, threshold, slice, bootstrap, and permutation-importance reports for
error analysis. Top permutation feature in this run: `{top_feature}`.
"""
    path.write_text(card, encoding="utf-8")
    return path


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def run(args: argparse.Namespace) -> dict:
    validate_runtime_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)

    raw_path = Path(args.data_dir) / "winequality-red.csv"
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    status(f"Using device: {device}", args.quiet)
    status("Downloading or reusing the raw dataset", args.quiet)
    download_dataset(raw_path, url=args.url, force=args.force_download)
    raw_frame = load_raw_data(raw_path)
    status(f"Loaded {len(raw_frame)} rows and {len(FEATURE_COLUMNS)} features", args.quiet)
    cleaned_frame = clean_and_label(
        raw_frame,
        quality_threshold=args.quality_threshold,
    )
    status(f"Prepared {len(cleaned_frame)} clean rows", args.quiet)
    save_json(
        {
            "raw": summarize_dataset(raw_frame),
            "cleaned": summarize_dataset(cleaned_frame),
            "quality_threshold": args.quality_threshold,
        },
        report_dir / "data_quality.json",
    )

    train_frame, validation_frame, test_frame = split_dataset(
        cleaned_frame,
        seed=args.seed,
    )
    status(
        "Split data into "
        f"{len(train_frame)} train / {len(validation_frame)} validation / "
        f"{len(test_frame)} test rows",
        args.quiet,
    )
    split_shift = {
        "validation": feature_shift_report(train_frame, validation_frame),
        "test": feature_shift_report(train_frame, test_frame),
    }
    save_json(split_shift, report_dir / "split_shift.json")

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_frame[FEATURE_COLUMNS]).astype(
        np.float32
    )
    validation_features = scaler.transform(
        validation_frame[FEATURE_COLUMNS]
    ).astype(np.float32)
    test_features = scaler.transform(test_frame[FEATURE_COLUMNS]).astype(np.float32)
    save_scaler(scaler, FEATURE_COLUMNS, artifact_dir / "scaler.json")

    train_labels = train_frame["good_quality"].to_numpy(dtype=np.float32)
    validation_labels = validation_frame["good_quality"].to_numpy(dtype=np.float32)
    negative_count = float((train_labels == 0).sum())
    positive_count = float((train_labels == 1).sum())
    positive_weight = negative_count / max(positive_count, 1.0)

    train_loader = make_loader(
        train_features,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_features,
        validation_labels,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TabularMLP(input_dim=len(FEATURE_COLUMNS))
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
    )
    model, history, training_summary = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        positive_weight=positive_weight,
        config=train_config,
        device=device,
    )
    status(
        f"Training stopped after {training_summary['epochs_trained']} epochs",
        args.quiet,
    )

    selected_threshold = args.threshold
    threshold_report = None
    threshold_stability = None
    if args.tune_threshold:
        validation_probabilities = predict_probabilities(
            model,
            validation_features,
            device,
        )
        threshold_report = threshold_search(
            validation_labels.astype(int),
            validation_probabilities,
            metric=args.threshold_metric,
        )
        selected_threshold = float(threshold_report["best_threshold"])
        save_json(threshold_report, report_dir / "threshold_search.json")
        save_threshold_curve(
            threshold_report["thresholds"],
            args.threshold_metric,
            report_dir / "threshold_curve.png",
        )
        if args.bootstrap_samples > 0:
            threshold_stability = bootstrap_threshold_stability(
                validation_labels.astype(int),
                validation_probabilities,
                metric=args.threshold_metric,
                samples=args.bootstrap_samples,
                seed=args.seed + 11,
            )
            save_json(
                threshold_stability,
                report_dir / "threshold_stability.json",
            )
        status(
            f"Selected threshold {selected_threshold:.3f} "
            f"by validation {args.threshold_metric}",
            args.quiet,
        )

    pd.DataFrame(history).to_csv(artifact_dir / "training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_COLUMNS,
            "quality_threshold": args.quality_threshold,
            "classification_threshold": selected_threshold,
            "training_summary": training_summary,
        },
        artifact_dir / "model.pt",
    )

    metrics = save_evaluation_outputs(
        model=model,
        test_features=test_features,
        test_frame=test_frame,
        history=history,
        output_dir=report_dir,
        device=device,
        threshold=selected_threshold,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        slice_features=tuple(args.slice_features),
        slice_bins=args.slice_bins,
        robustness_repeats=args.robustness_repeats,
        mc_dropout_passes=args.mc_dropout_passes,
    )
    importance_report = None
    if args.permutation_importance_repeats > 0:
        importance_report = permutation_importance(
            model=model,
            features=test_features,
            labels=test_frame["good_quality"].to_numpy(dtype=int),
            feature_names=FEATURE_COLUMNS,
            device=device,
            threshold=selected_threshold,
            metric=args.importance_metric,
            repeats=args.permutation_importance_repeats,
            seed=args.seed + 17,
        )
        save_json(importance_report, report_dir / "permutation_importance.json")
        pd.DataFrame(importance_report["features"]).to_csv(
            report_dir / "permutation_importance.csv",
            index=False,
        )
        save_permutation_importance_plot(
            importance_report["features"],
            report_dir / "permutation_importance.png",
        )
    roc_auc = metrics["roc_auc"]
    roc_auc_text = f"{roc_auc:.3f}" if roc_auc is not None else "n/a"
    status(
        f"Test accuracy: {metrics['accuracy']:.3f} | "
        f"ROC-AUC: {roc_auc_text}",
        args.quiet,
    )

    run_summary = {
        "dataset_url": args.url,
        "device": str(device),
        "seed": args.seed,
        "split_sizes": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "split_shift": split_shift,
        "feature_count": len(FEATURE_COLUMNS),
        "quality_threshold": args.quality_threshold,
        "requested_threshold": args.threshold,
        "selected_threshold": selected_threshold,
        "threshold_tuning": threshold_report,
        "threshold_stability": threshold_stability,
        "permutation_importance": importance_report,
        "training": training_summary,
        "test_metrics": metrics,
    }
    save_json(run_summary, report_dir / "run_summary.json")
    write_model_card(run_summary, report_dir / "model_card.md")
    (artifact_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete wine-quality PyTorch pipeline."
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--url", default=DATA_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--quality-threshold", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--permutation-importance-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--robustness-repeats", type=int, default=5)
    parser.add_argument("--mc-dropout-passes", type=int, default=30)
    parser.add_argument(
        "--slice-features",
        nargs="*",
        default=["alcohol", "volatile acidity", "sulphates"],
    )
    parser.add_argument("--slice-bins", type=int, default=3)
    parser.add_argument(
        "--importance-metric",
        choices=["f1", "balanced_accuracy", "matthews_correlation", "roc_auc", "average_precision"],
        default="f1",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "balanced_accuracy", "matthews_correlation"],
        default="f1",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
