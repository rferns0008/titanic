from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TITLE_MAPPING = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 2,
    'Master': 3,
    'Rare': 4,
    'Unknown': 4,
}

EMBARKED_MAPPING = {'S': 0, 'C': 1, 'Q': 2}
SEX_MAPPING = {'male': 0, 'female': 1}


def extract_title(name: str | float) -> str:
    if pd.isna(name):
        return 'Unknown'

    match = re.search(r',\s*([^\.]+)\.', str(name))
    if not match:
        return 'Unknown'

    return match.group(1).strip()


def load_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path)


def preprocess_features(df: pd.DataFrame, training: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    df = df.copy()

    target = None
    if training:
        target = df['Survived'].copy()

    df['Title'] = df['Name'].apply(extract_title)
    df['Title'] = df['Title'].replace(
        {
            'Mlle': 'Miss',
            'Ms': 'Miss',
            'Mme': 'Mrs',
            'Lady': 'Rare',
            'Countess': 'Rare',
            'Sir': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Capt': 'Rare',
        }
    )

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(
        df['Embarked'].mode().iloc[0] if not df['Embarked'].mode().empty else 'S'
    )

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Sex'] = df['Sex'].map(SEX_MAPPING).fillna(0).astype(int)
    df['Embarked'] = df['Embarked'].map(EMBARKED_MAPPING).fillna(0).astype(int)
    df['Title'] = df['Title'].map(TITLE_MAPPING).fillna(TITLE_MAPPING['Unknown']).astype(int)

    feature_columns = [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked',
        'FamilySize',
        'IsAlone',
        'Title',
    ]

    X = df[feature_columns]
    return X, target


def save_processed_data(df: pd.DataFrame, file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
