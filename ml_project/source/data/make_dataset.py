from typing import Tuple

import pandas as pd
import sklearn

from source.entities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = sklearn.model_selection.train_test_split(
        data, test_size=params.validation_size, random_state=params.random_state
    )
    return train_data, test_data
