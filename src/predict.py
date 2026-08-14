from __future__ import annotations

import argparse
import pickle
from pathlib import Path

try:
    from src.preprocess import preprocess_features
except ModuleNotFoundError:  # pragma: no cover - compatibility for direct script execution
    from preprocess import preprocess_features


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def predict_single_record(model_path: Path, record: dict) -> dict:
    # Children aged 0-15 are predicted to have survived
    age = record.get('Age')
    if age is not None and 0 <= age <= 15:
        return {
            'prediction': 1,
            'prediction_label': 'Survived',
            'probability': {
                'did_not_survive': 0.0,
                'survived': 1.0,
            },
            'confidence': 1.0,
            'note': 'Children aged 0-15 are predicted to have survived (historical context: children had priority in lifeboats)',
        }

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    df = __import__('pandas').DataFrame([record])
    X, _ = preprocess_features(df, training=False)
    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X)[0]

    return {
        'prediction': prediction,
        'prediction_label': 'Survived' if prediction == 1 else 'Did not survive',
        'probability': {
            'did_not_survive': float(probability[0]),
            'survived': float(probability[1]),
        },
        'confidence': float(max(probability)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict survival for one Titanic passenger record.')
    parser.add_argument(
        '--model-path',
        type=Path,
        default=WORKSPACE_ROOT / 'models' / 'titanic_model.pkl',
        help='Path to the trained model pickle.',
    )
    parser.add_argument(
        '--record',
        type=str,
        required=True,
        help='JSON string containing passenger features.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import json

    record = json.loads(args.record)
    result = predict_single_record(args.model_path, record)
    print(result)


if __name__ == '__main__':
    main()
