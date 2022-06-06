import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from typing import Union
import joblib


def predict_model(
    model: Union[LogisticRegression, XGBClassifier], features: pd.DataFrame
) -> np.ndarray:
    return model.predict(features)


def load_model(path: str) -> Union[LogisticRegression, XGBClassifier]:
    model = joblib.load(path)
    return model


def load_transformer(path: str) -> ColumnTransformer:
    transformer = joblib.load(path)
    return transformer
