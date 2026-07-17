# Wine Quality Modeling Pipeline

PyTorch classification pipeline for estimating whether a red wine has a UCI quality score of 6 or higher from its physicochemical measurements.

## Project summary

The pipeline downloads the UCI red-wine dataset, checks its schema, cleans the records, creates a binary target, and produces stratified train, validation, and test sets. Standardization is fitted on the training data only. The trained model and evaluation artifacts are saved locally so a run can be inspected or repeated without rebuilding the workflow.

The original `quality` score is used to create `good_quality`, but it is not passed to the model. This keeps the target separate from the 11 input features.

## Approach

1. Validate the expected numeric columns in the semicolon-delimited CSV.
2. Remove non-finite, incomplete, and duplicate records.
3. Create `good_quality` using `quality >= 6`.
4. Make deterministic stratified 60/20/20 train, validation, and test splits.
5. Fit `StandardScaler` on the training partition and apply it to the other partitions.
6. Train and evaluate the PyTorch model.

The model is a small `TabularMLP` with 11 inputs, hidden layers of 64 and 32 units, ReLU activations, dropout, and one output logit. Training uses Adam, class-weighted `BCEWithLogitsLoss`, and early stopping based on validation loss. Evaluation includes accuracy, precision, recall, F1, balanced accuracy, Matthews correlation coefficient, ROC-AUC, and average precision.

## Technology

Python, pandas, NumPy, scikit-learn, PyTorch, Matplotlib, pytest, and GitHub Actions.

## Quick start

```bash
git clone https://github.com/amir-sbg/wine-quality-pytorch-pipeline.git
cd wine-quality-pytorch-pipeline

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
```

The dataset is downloaded to `data/raw/` on the first run. Model files are written to `artifacts/`, and metrics, plots, and test predictions are written to `reports/`.

Training settings are configurable, for example:

```bash
python -m src.pipeline \
  --epochs 120 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --quality-threshold 6 \
  --seed 42 \
  --device auto
```

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
├── run_summary.json
├── test_predictions.csv
├── confusion_matrix.png
├── learning_curve.png
└── roc_curve.png
```

`test_predictions.csv` includes the test rows, predicted probabilities, class predictions, and a correctness flag for reviewing errors.

## Project structure

```text
.
├── src/
│   ├── data.py          # loading, validation, cleaning, and profiling
│   ├── model.py         # PyTorch tabular model
│   ├── train.py         # data loaders, loss, and early stopping
│   ├── evaluate.py      # metrics, plots, and predictions
│   └── pipeline.py      # command-line workflow
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

The dataset is small, and the quality threshold is an explicit modeling choice. The saved reports make the split, preprocessing, training settings, and test errors available for comparison.
