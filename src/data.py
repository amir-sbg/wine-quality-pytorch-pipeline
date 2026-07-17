from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
LABEL_COLUMN = "good_quality"


def download_dataset(destination: Path, url: str = DATA_URL, force: bool = False) -> Path:
    """Download the raw UCI CSV once and return its local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    request = Request(url, headers={"User-Agent": "datascience-pipline/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    return destination


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the semicolon-delimited UCI file and validate its schema."""
    frame = pd.read_csv(path, sep=";")
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Fail early when the downloaded file does not match the expected dataset."""
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric columns, found: {non_numeric}")


def clean_and_label(
    frame: pd.DataFrame,
    quality_threshold: int = 6,
) -> pd.DataFrame:
    """Clean numeric records and create the documented binary target.

    A wine is labeled as good quality when its original score is at least six.
    The original score remains in the returned frame for analysis, but it is
    never included in the model feature matrix.
    """
    cleaned = frame.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned[LABEL_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(int) >= quality_threshold
    ).astype(int)
    return cleaned


def summarize_dataset(frame: pd.DataFrame) -> dict:
    """Return a JSON-serializable data-quality summary."""
    missing = frame.isna().sum().astype(int).to_dict()
    target_counts = (
        frame[LABEL_COLUMN].value_counts().sort_index().astype(int).to_dict()
        if LABEL_COLUMN in frame
        else {}
    )
    numeric_summary = frame[FEATURE_COLUMNS].describe().round(4).to_dict()
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": missing,
        "target_counts": {str(key): value for key, value in target_counts.items()},
        "feature_summary": numeric_summary,
    }


def split_dataset(
    frame: pd.DataFrame,
    seed: int,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, stratified train, validation, and test frames.

    ``validation_size`` and ``test_size`` are expressed as fractions of the
    complete input frame. Stratification is performed on the derived binary
    target so each partition keeps a comparable class balance.
    """
    if LABEL_COLUMN not in frame:
        raise ValueError(f"Dataset must contain the target column: {LABEL_COLUMN}")
    if not (0 < validation_size < 1 and 0 < test_size < 1):
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must sum to less than 1")

    holdout_size = validation_size + test_size
    train_frame, holdout_frame = train_test_split(
        frame,
        test_size=holdout_size,
        stratify=frame[LABEL_COLUMN],
        random_state=seed,
    )
    test_fraction_of_holdout = test_size / holdout_size
    validation_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=test_fraction_of_holdout,
        stratify=holdout_frame[LABEL_COLUMN],
        random_state=seed,
    )
    return tuple(
        partition.reset_index(drop=True)
        for partition in (train_frame, validation_frame, test_frame)
    )


def save_json(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
