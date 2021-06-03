import os

import click
from sklearn.datasets import load_wine


@click.command("download")
@click.option("--output-dir")
def download(output_dir: str):
    x, y = load_wine(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)
    x.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)
    #x.to_csv('/'.join([output_dir, "data.csv"]), index=False)
    #y.to_csv('/'.join([output_dir, "target.csv"]), index=False)


if __name__ == '__main__':
    download()
