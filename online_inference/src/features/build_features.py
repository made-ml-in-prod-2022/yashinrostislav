import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.entities.feature_params import FeatureParams


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline([("standard_scaler", StandardScaler())])
    return numerical_pipeline


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline([("ohe", OneHotEncoder())])
    return categorical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    transformer._features = params.categorical_features + params.numerical_features
    return transformer


def serialize_transformer(transformer: ColumnTransformer, fout: str) -> None:
    with open(fout, "wb") as f:
        joblib.dump(transformer, f)


def process_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df[transformer._features]))


def extract_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    return df[params.categorical_features + params.numerical_features]


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]

