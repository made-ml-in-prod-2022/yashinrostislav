import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split
from typing import NoReturn

TEST_SIZE = 0.2
TRAIN_SIZE = 0.75


@click.command()
@click.option("--input-dir", required=True)
def split_data(input_dir: str) -> NoReturn:
    X = pd.read_csv(os.path.join(input_dir, "X.csv"))
    y = pd.read_csv(os.path.join(input_dir, "y.csv"))

    X_train_val, X_test, y_train_val, _ = train_test_split(X, y, test_size=TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=TRAIN_SIZE
    )

    X_train.to_csv(os.path.join(input_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(input_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(input_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(input_dir, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(input_dir, "X_test.csv"), index=False)
