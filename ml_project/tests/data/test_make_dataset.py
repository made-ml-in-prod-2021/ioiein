import pandas as pd

from src.data import read_data, split_train_val_data
from src.entities import SplittingParams


def test_read_data(input_data_path: str):
    data = read_data(input_data_path)
    assert isinstance(data, pd.DataFrame)


def test_split_train_val_data(input_data_path: str):
    data = read_data(input_data_path)
    params = SplittingParams(validation_size=0.2, random_state=225)
    train_data, val_data = split_train_val_data(data, params)
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(val_data, pd.DataFrame)
    assert len(train_data) + len(val_data) == len(data)