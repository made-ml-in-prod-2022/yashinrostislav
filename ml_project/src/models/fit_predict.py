import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.compose import ColumnTransformer
from typing import Dict, Union
import joblib
import json

from src.entities.train_params import TrainingParams


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> Union[LogisticRegression, XGBClassifier]:
    if train_params.model_type == "LogisticRegression":
        model = LogisticRegression(**train_params.model_params)
    elif train_params.model_type == "XGBClassifier":
        model = XGBClassifier(**train_params.model_params)
    model.fit(features, target)
    return model


def predict_model(
    model: Union[LogisticRegression, XGBClassifier], features: pd.DataFrame
) -> np.ndarray:
    return model.predict(features)


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "f1_score": f1_score(target, predicts, average="weighted"),
        "balanced_accuracy_score": balanced_accuracy_score(target, predicts),
    }


def serialize_model(model: Union[LogisticRegression, XGBClassifier], fout: str) -> None:
    with open(fout, mode="wb") as f:
        joblib.dump(model, f)


def load_model(path: str) -> Union[LogisticRegression, XGBClassifier]:
    model = joblib.load(path)
    return model


def load_transformer(path: str) -> ColumnTransformer:
    transformer = joblib.load(path)
    return transformer


def save_preds(preds: np.ndarray, fout: str) -> None:
    with open(fout, mode="w") as f:
        json.dump(preds.tolist(), f)
