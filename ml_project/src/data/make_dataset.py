import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.entities.split_params import SplittingParams


def read_data(input_data_path: str) -> pd.DataFrame:
    return pd.read_csv(input_data_path)


def split_train_val_data(
    data: pd.DataFrame, splitting_params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train, data_val = train_test_split(
        data,
        train_size=splitting_params.train_size,
        random_state=splitting_params.random_state,
    )
    return data_train, data_val
