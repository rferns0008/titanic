# MLOps Titanic Training

This repository contains a simple Titanic machine learning pipeline using the workspace data layout.

## Repository structure

- `data/train.csv` - training dataset
- `data/test.csv` - test dataset for predictions
- `src/preprocess.py` - preprocessing utilities
- `src/train.py` - training and prediction script
- `models/` - output folder for saved model artifacts
- `outputs/` - output folder for prediction CSV files
- `requirements.txt` - Python dependencies

## Setup

1. Open a terminal in the workspace root:
   ```bash
   cd /Users/rahulfernandes/Documents/MLOps
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

## Train the model

Run the training script to fit the model and save it as a pickle file:

```bash
python3 ./src/train.py
```

The default trained model is saved to:

- `models/titanic_model.pkl`

## Predict on test data

To generate predictions using `data/test.csv` and save the output CSV:

```bash
python3 ./src/train.py --test-csv data/test.csv
```

The default prediction output is saved to:

- `outputs/test_predictions.csv`

## Custom paths

You can override paths using command-line arguments:

```bash
python3 ./src/train.py \
  --train-csv data/train.csv \
  --model-output models/titanic_model.pkl \
  --test-csv data/test.csv \
  --predictions-output outputs/test_predictions.csv
```

## Notes

- The training pipeline uses a `RandomForestClassifier`.
- Preprocessing includes title extraction, missing value handling, encoding, and feature construction.
- Ensure the virtual environment is activated before running scripts.

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
export AWS_ACCESS_KEY_ID="YOUR_B2_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_B2_APP_KEY"
export AWS_DEFAULT_REGION="your-region-code"
```

Helper script
- Use the provided helper to pull data and run training. It will source `.venv.nosync` if present, run `dvc pull`, then run the training script:

```bash
./scripts/run_train.sh --train-csv data/train.csv --model-output models/titanic_model.pkl
```

You can skip the automatic `dvc pull` by setting `SKIP_DVC_PULL=1` in the environment.

