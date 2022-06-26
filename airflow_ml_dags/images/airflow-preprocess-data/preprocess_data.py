import os
import pandas as pd
import click
from typing import NoReturn


@click.command()
@click.option("--input-dir", required=True)
@click.option("--output-dir", required=True)
def preprocess_data(input_dir: str, output_dir: str) -> NoReturn:
    X_raw = pd.read_csv(os.path.join(input_dir, "X.csv"))
    y_raw = pd.read_csv(os.path.join(input_dir, "y.csv"))

    Xy = X_raw.copy()
    Xy["target"] = y_raw.values
    Xy.dropna(inplace=True)

    X_preprocessed = Xy.drop(columns="target")
    y_preprocessed = Xy["target"].copy()

    os.makedirs(output_dir, exist_ok=True)
    X_preprocessed.to_csv(os.path.join(output_dir, "X.csv"), index=False)
    y_preprocessed.to_csv(os.path.join(output_dir, "y.csv"), index=False)
