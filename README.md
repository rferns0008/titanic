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
