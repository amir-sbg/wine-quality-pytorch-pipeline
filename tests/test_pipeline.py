import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    FEATURE_COLUMNS,
    clean_and_label,
    split_dataset,
    validate_schema,
)
from src.evaluate import classification_metrics
from src.model import TabularMLP
from src.train import TrainConfig, make_loader, train_model


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


def test_train_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=0)


def test_metrics_include_balanced_views() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["matthews_correlation"] == 1.0


def test_metrics_reject_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        classification_metrics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            threshold=1.5,
        )


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
