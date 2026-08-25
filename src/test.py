from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

try:
    from src.preprocess import load_data, preprocess_features
except ModuleNotFoundError:  # pragma: no cover - compatibility for direct script execution
    from preprocess import load_data, preprocess_features


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def evaluate_model(test_csv: Path, model_path: Path, output_path: Path) -> pd.DataFrame:
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
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate predictions for a Titanic test set.')
    parser.add_argument(
        '--test-csv',
        type=Path,
        required=True,
        help='Path to the test CSV file.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=WORKSPACE_ROOT / 'models' / 'titanic_model.pkl',
        help='Path to the trained model pickle.',
    )
    parser.add_argument(
        '--predictions-output',
        type=Path,
        default=WORKSPACE_ROOT / 'outputs' / 'test_predictions.csv',
        help='Where to write predictions output.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(args.test_csv, args.model_path, args.predictions_output)


if __name__ == '__main__':
    main()
