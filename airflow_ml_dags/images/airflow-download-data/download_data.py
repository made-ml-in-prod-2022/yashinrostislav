import os
import click
from sklearn.datasets import load_wine
from typing import NoReturn


@click.command()
@click.option("--output-dir", required=True)
def download_data(output_dir: str) -> NoReturn:
    X, y = load_wine(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "X.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y.csv"), index=False)
