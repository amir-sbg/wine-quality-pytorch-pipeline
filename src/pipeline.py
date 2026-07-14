from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import (
    DATA_URL,
    FEATURE_COLUMNS,
    clean_and_label,
    download_dataset,
    load_raw_data,
    save_json,
    summarize_dataset,
)
from .evaluate import save_evaluation_outputs
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
    return torch.device(requested)


def save_scaler(scaler: StandardScaler, feature_names: list[str], path: Path) -> None:
    payload = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    save_json(payload, path)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)

    raw_path = Path(args.data_dir) / "winequality-red.csv"
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(raw_path, url=args.url, force=args.force_download)
    raw_frame = load_raw_data(raw_path)
    cleaned_frame = clean_and_label(
        raw_frame,
        quality_threshold=args.quality_threshold,
    )
    save_json(
        {
            "raw": summarize_dataset(raw_frame),
            "cleaned": summarize_dataset(cleaned_frame),
            "quality_threshold": args.quality_threshold,
        },
        report_dir / "data_quality.json",
    )

    train_frame, holdout_frame = train_test_split(
        cleaned_frame,
        test_size=0.40,
        stratify=cleaned_frame["good_quality"],
        random_state=args.seed,
    )
    validation_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=0.50,
        stratify=holdout_frame["good_quality"],
        random_state=args.seed,
    )

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
    )
    validation_loader = make_loader(
        validation_features,
        validation_labels,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TabularMLP(input_dim=len(FEATURE_COLUMNS))
    train_config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    model, history, training_summary = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        positive_weight=positive_weight,
        config=train_config,
        device=device,
    )
    pd.DataFrame(history).to_csv(artifact_dir / "training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_COLUMNS,
            "quality_threshold": args.quality_threshold,
            "classification_threshold": args.threshold,
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
        threshold=args.threshold,
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
        "feature_count": len(FEATURE_COLUMNS),
        "quality_threshold": args.quality_threshold,
        "training": training_summary,
        "test_metrics": metrics,
    }
    save_json(run_summary, report_dir / "run_summary.json")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
