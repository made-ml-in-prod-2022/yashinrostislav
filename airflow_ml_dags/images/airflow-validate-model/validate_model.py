import os
import pandas as pd
import click
import pickle
from sklearn.metrics import f1_score
from typing import NoReturn


@click.command()
@click.option("--model-dir", required=True)
@click.option("--data-dir", required=True)
def validate_model(model_dir: str, data_dir: str) -> NoReturn:
    with open(os.path.join(model_dir, "model.pkl"), mode="rb") as f:
        model = pickle.load(f)

    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv"))

    score = f1_score(y_val, model.predict(X_val))

    with open(os.path.join(model_dir, "f1_score.txt")) as f:
        f.write(score)
