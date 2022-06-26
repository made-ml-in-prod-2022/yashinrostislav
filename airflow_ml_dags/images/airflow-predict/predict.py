import os
import pandas as pd
import click
import pickle
from typing import NoReturn


@click.command()
@click.option("--model-dir", required=True)
@click.option("--data-dir", required=True)
def predict(model_dir: str, data_dir: str) -> NoReturn:
    with open(os.path.join(model_dir, "model.pkl"), mode="rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))

    y_pred = pd.Series(model.predict(X_test))
    y_pred.to_csv(os.path.join(data_dir, "predictions.csv"), index=False)
