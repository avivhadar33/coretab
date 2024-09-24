import pandas as pd
from pandas.api.types import is_numeric_dtype
from dataclasses import dataclass
from types import FunctionType


def encode_categorical_features(df):
    categorical_columns = []
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            categorical_columns.append(c)

    for c in categorical_columns:
        print(c)
        one_hot = pd.get_dummies(df[c], prefix=c, dtype=int)
        one_hot.columns = [c.replace(' ', '_').replace(',', '-').replace('<', '(').replace('>', ')') for c in
                           one_hot.columns]
        df = df.join(one_hot).drop(columns=c)
    return df


def null_columns(df):
    for c in df.columns:
        if any(df[c].isna().tolist()):
            df[c + '_null'] = df[c].isna().astype(int)
    df.fillna(0, inplace=True)
    return df


@dataclass
class Dataset:
    df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    target_col: str
    metric: FunctionType
