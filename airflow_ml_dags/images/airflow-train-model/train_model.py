import os
import pandas as pd
import click
from sklearn.linear_model import LogisticRegression
import pickle
from typing import NoReturn


@click.command()
@click.option("--input-dir", required=True)
@click.option("--model-dir", required=True)
def train_model(input_dir: str, model_dir: str) -> NoReturn:
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    model = LogisticRegression()
    model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), mode="wb") as f:
        pickle.dump(model, f)
