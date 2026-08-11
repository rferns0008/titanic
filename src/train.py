from __future__ import annotations

import argparse
from pathlib import Path

import pickle
import subprocess
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import load_data, preprocess_features


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def build_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=random_state,
        n_jobs=-1,
    )


def train_model(train_csv: Path, model_path: Path, random_state: int = 42) -> None:
    df = load_data(train_csv)
    X, y = preprocess_features(df, training=True)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    print('Training complete')
    print(f'Model saved to: {model_path}')
    print(f'Validation accuracy: {accuracy:.4f}')
    print('\nClassification report:')
    print(classification_report(y_valid, y_pred, digits=4))


def predict(test_csv: Path, model_path: Path, output_path: Path) -> None:
    df = load_data(test_csv)
    X, _ = preprocess_features(df, training=False)

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    predictions = model.predict(X)

    result = pd.DataFrame(
        {
            'PassengerId': df['PassengerId'],
            'Survived': predictions.astype(int),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'Predictions written to: {output_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a Titanic classifier and optionally score a test set.'
    )
    parser.add_argument(
        '--train-csv',
        type=Path,
        default=WORKSPACE_ROOT / 'data' / 'train.csv',
        help='Path to the training CSV file.',
    )
    parser.add_argument(
        '--model-output',
        type=Path,
        default=WORKSPACE_ROOT / 'models' / 'titanic_model.pkl',
        help='Where to save the trained model.',
    )
    parser.add_argument(
        '--test-csv',
        type=Path,
        default=None,
        help='Optional path to a test CSV file for prediction.',
    )
    parser.add_argument(
        '--predictions-output',
        type=Path,
        default=WORKSPACE_ROOT / 'outputs' / 'test_predictions.csv',
        help='Where to write test predictions if --test-csv is provided.',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Attempt to pull DVC-tracked data from configured remote before training.
    # Set environment variable `SKIP_DVC_PULL=1` to skip this behavior (useful for tests).
    if not os.environ.get('SKIP_DVC_PULL'):
        try:
            subprocess.run(["dvc", "pull"], check=True)
        except FileNotFoundError:
            print("dvc executable not found in PATH; skipping 'dvc pull'.")
        except subprocess.CalledProcessError:
            print("'dvc pull' returned a non-zero exit code; continuing anyway.")

    train_model(args.train_csv, args.model_output, random_state=args.random_state)

    if args.test_csv is not None:
        predict(args.test_csv, args.model_output, args.predictions_output)


if __name__ == '__main__':
    main()
