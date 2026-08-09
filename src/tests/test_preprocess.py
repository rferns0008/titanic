import pandas as pd

from preprocess import extract_title, preprocess_features


def test_extract_title_handles_missing_name():
    assert extract_title(float('nan')) == 'Unknown'


def test_extract_title_parses_title():
    assert extract_title('Doe, Mr. John') == 'Mr'


def test_preprocess_features_training():
    df = pd.DataFrame(
        {
            'PassengerId': [1, 2],
            'Survived': [0, 1],
            'Pclass': [3, 1],
            'Name': ['Smith, Mr. John', 'Doe, Mrs. Jane'],
            'Sex': ['male', 'female'],
            'Age': [22.0, None],
            'SibSp': [1, 0],
            'Parch': [0, 1],
            'Fare': [7.25, None],
            'Embarked': ['S', None],
        }
    )

    X, y = preprocess_features(df, training=True)

    assert y.tolist() == [0, 1]
    assert 'Title' in X.columns
    assert X.shape == (2, 10)
