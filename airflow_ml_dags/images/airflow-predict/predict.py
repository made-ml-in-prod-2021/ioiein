import os
import pickle
from pathlib import Path

import click
import pandas as pd


@click.command("predict")
@click.option("--input-dir")
@click.option("--input-model-dir")
@click.option("--output-dir")
def predict(input_dir: str, input_model_dir: str, output_dir: str):
    input_data_path = Path(input_dir)
    input_model_path = Path(input_model_dir)
    output_dir_path = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_model_path / "model", "rb") as f:
        model = pickle.load(f)

    data = pd.read_csv(input_data_path / "data.csv")

    predicts = model.predict(data)

    predicts_data = pd.DataFrame(predicts, columns=["target"])

    output_dir_path.mkdir(parents=True, exist_ok=True)
    predicts_data.to_csv(output_dir_path / "predicts.csv", index=False)


if __name__ == "__main__":
    predict()
