# MLOps Titanic Training

This repository contains a simple Titanic machine learning pipeline using a package-oriented Python layout.

## Repository structure

- `data/train.csv` - training dataset
- `data/test.csv` - test dataset for predictions
- `src/__init__.py` - package marker for the project source package
- `src/preprocess.py` - preprocessing utilities
- `src/train.py` - training pipeline entry point
- `src/predict.py` - prediction entry point
- `src/tests/test_preprocess.py` - unit tests for preprocessing logic
- `examples/sample_predictions.csv` - small sample artifact for documentation and demos
- `models/` - output folder for saved model artifacts
- `outputs/` - output folder for prediction CSV files
- `requirements.txt` - Python dependencies
- `pytest.ini` - pytest configuration for repo-root execution

## Setup

1. Open a terminal in the workspace root:
   ```bash
   cd /Users/rahulfernandes/projects/titanic
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv.nosync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv.nosync/bin/activate
   ```

4. Install dependencies:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

## Run tests

Run the project test suite from the repository root:

```bash
python -m pytest -v
```

To run a specific test file:

```bash
python -m pytest src/tests/test_preprocess.py -v
```

## Train the model

Run the training script from the repository root:

```bash
python src/train.py
```

The default trained model is saved to:

- `models/titanic_model.pkl`

## Predict on test data

To generate predictions using `data/test.csv` and save the output CSV:

```bash
python src/predict.py --model-path models/titanic_model.pkl --test-csv data/test.csv --predictions-output outputs/test_predictions.csv
```

The default prediction output is saved to:

- `outputs/test_predictions.csv`

## Real-time prediction

To make predictions on individual passenger records, use the `predict_single_record` function.

### Method 1: Python API

```python
from src.predict import predict_single_record
from pathlib import Path

# Define a passenger record
record = {
    "Name": "Doe, Miss. Jane",
    "Pclass": 1,
    "Sex": "female",
    "Age": 25.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 512.33,
    "Embarked": "S"
}

# Make prediction
result = predict_single_record(
    model_path=Path("models/titanic_model.pkl"),
    record=record
)

print(result)
```

Expected output:
```python
{
    "prediction": 1,
    "prediction_label": "Survived",
    "probability": {
        "did_not_survive": 0.15,
        "survived": 0.85
    },
    "confidence": 0.85
}
```

### Method 2: Command Line

Use the CLI with a JSON string containing passenger features:

```bash
python src/predict.py \
  --model-path models/titanic_model.pkl \
  --record "{"Name": "Doe, Miss. Jane", "Pclass": 1, "Sex": "female", "Age": 25.0, "SibSp": 1, "Parch": 0, "Fare": 512.33, "Embarked": "S"}"
```

This will output the prediction result, probability distribution, and confidence score for the single passenger record.

### Required fields for prediction

The following fields are **required** for making predictions:
- `Name" - Passenger name (format: "Last, Title. First" e.g., "Doe, Miss. Jane")
- `Pclass` - Passenger class (1, 2, or 3)
- `Sex` - Gender ("male" or "female")
- `Age` - Passenger age (numeric)
- `SibSp` - Number of siblings/spouses aboard (numeric)
- `Parch` - Number of parents/children aboard (numeric)
- `Fare` - Ticket fare (numeric)
- `Embarked` - Port of embarkation ("S", "C", or "Q")

### Special rule: Children (ages 0-15)

Children aged 0-15 years are **always predicted to have survived**, regardless of other features. This reflects the historical priority given to children in lifeboats during the Titanic disaster.

Example:
```bash
python src/predict.py \
  --model-path models/titanic_model.pkl \
  --record "{"Name": "Smith, Master. Thomas", "Pclass": 3, "Sex": "male", "Age": 8.0, "SibSp": 1, "Parch": 1, "Fare": 50.0, "Embarked": "S"}"
```

Output:
```
{"prediction": 1, "prediction_label": "Survived", "probability": {"did_not_survive": 0.0, "survived": 1.0}, "confidence": 1.0, "note": "Children aged 0-15 are predicted to have survived (historical context: children had priority in lifeboats)"}
```

## Custom paths

You can override paths using command-line arguments:

```bash
python src/train.py \
  --train-csv data/train.csv \
  --model-output models/titanic_model.pkl
```

## Package import pattern

The project is configured to use a package-aware import pattern:

```python
from src.preprocess import load_data, preprocess_features
```

This keeps imports working reliably from the repository root and in pytest. A compatibility fallback is also present for direct script execution when needed.

## Documentation sample artifact

A small, versioned example output is kept under [examples/sample_predictions.csv](examples/sample_predictions.csv). This is intended for documentation and demo purposes only and should not be confused with generated runtime artifacts in [models](models) or [outputs](outputs).

## Artifact types and purpose

- [data](data) contains the source dataset managed by DVC and should remain available to the training pipeline.
- [examples](examples) contains small, checked-in sample files used for documentation and illustration.
- [models](models) and [outputs](outputs) are generated runtime artifacts and are intentionally ignored by Git.

This separation keeps the repo clean while preserving a simple example for readers and maintainers.

## Notes

- The training pipeline uses a `RandomForestClassifier`.
- Preprocessing includes title extraction, missing value handling, encoding, and feature construction.
- Ensure the virtual environment is activated before running scripts or tests.

## DVC + Backblaze B2 (S3-compatible) integration

This project uses DVC to manage large data files. In CI/CD and local runs we use Backblaze B2 via its S3-compatible API.

CI requirements
- Add the following GitHub repository secrets:
   - `B2_KEY_ID` — Backblaze S3-compatible application Key ID
   - `B2_APP_KEY` — Backblaze application Key (secret)
   - `B2_REGION` — Backblaze region code (e.g. `us-west-000`, `eu-central-003`)

The GitHub workflows are configured to install `dvc[s3]` and run `dvc pull` using these secrets before tests and CD steps.

Local development
- Recommended: create and activate the virtualenv in the repo root:

```bash
python3 -m venv .venv.nosync
source .venv.nosync/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install "dvc[s3]"
```

- Export your Backblaze S3-compatible credentials in the shell where you run training:

```bash
export AWS_ACCESS_KEY_ID="003219d1b4ac7a60000000001"
export AWS_SECRET_ACCESS_KEY="K003qb9nnXFSW8zl+CEJGPVSxwrmb54"
export AWS_DEFAULT_REGION="eu-central-003"
```

Helper script
- Use the provided helper to pull data and run training. It will source `.venv.nosync` if present, run `dvc pull`, then run the training script:

```bash
./scripts/run_train.sh --train-csv data/train.csv --model-output models/titanic_model.pkl
```

You can skip the automatic `dvc pull` by setting `SKIP_DVC_PULL=1` in the environment.

