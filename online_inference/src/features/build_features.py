import pandas as pd
from sklearn.compose import ColumnTransformer


def process_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df[transformer._features]))
