# Wine Quality PyTorch Pipeline

A reproducible tabular-classification pipeline for predicting whether a red wine receives a UCI quality score of at least 6 from 11 physicochemical measurements.

The project follows the full modeling path: schema checks, cleaning, a stratified split, train-only standardization, PyTorch training, threshold selection, calibration, and held-out error analysis. The original `quality` score creates the `good_quality` label but is never included as an input feature.

## Model and evaluation

The model is a small `TabularMLP` with 64/32 ReLU layers and dropout. Training uses Adam, class-weighted `BCEWithLogitsLoss`, gradient clipping, and validation-based early stopping.

Evaluation includes accuracy, precision, recall, F1, balanced accuracy, specificity, ROC-AUC, average precision, Matthews correlation, log loss, calibration slope/intercept, Brier decomposition, bootstrap intervals, selective risk, decision-curve net benefit, threshold operating points, feature slices, and permutation importance. Split diagnostics also report standardized mean shifts and population stability index (PSI). The saved predictions and reports make it possible to inspect mistakes instead of relying on one aggregate score.

## Setup

```bash
git clone https://github.com/amir-sbg/wine-quality-pytorch-pipeline.git
cd wine-quality-pytorch-pipeline
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Run the default experiment:

```bash
python -m src.pipeline
```

Useful options for a controlled run include:

```bash
python -m src.pipeline \
  --epochs 120 \
  --batch-size 64 \
  --quality-threshold 6 \
  --tune-threshold \
  --bootstrap-samples 250 \
  --slice-features alcohol "volatile acidity" sulphates \
  --permutation-importance-repeats 5 \
  --seed 42 \
  --device auto
```

The raw UCI data is downloaded to `data/raw/` on the first run. Models and preprocessing artifacts are saved under `artifacts/`; metrics, plots, predictions, and diagnostics are written to `reports/`.

## Outputs

```text
artifacts/
├── model.pt
├── scaler.json
├── training_history.csv
└── run_config.json

reports/
├── data_quality.json
├── metrics.json
├── test_predictions.csv
    ├── calibration.json
    ├── calibration_slope.json
    ├── brier_decomposition.json
    ├── selective_risk.json
    ├── decision_curve.json
    ├── operating_points.json
├── threshold_search.json
├── slice_metrics.json
├── permutation_importance.json
└── model_card.md
```

The notebooks in `notebooks/` provide an exploratory view of the same preprocessing and diagnostic outputs used by the command-line pipeline.

## Project layout

```text
src/
├── data.py       loading, validation, cleaning, and profiling
├── model.py      PyTorch tabular model
├── train.py      loaders, loss, and early stopping
├── evaluate.py   metrics, plots, and error analysis
└── pipeline.py   command-line workflow
tests/
```
