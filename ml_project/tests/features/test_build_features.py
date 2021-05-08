from typing import List
import pytest
import pandas as pd
import numpy as np
import sklearn.utils.validation

from src.data import read_data
from src.features import extract_target, build_transformer, make_features
from src.entities import FeatureParams
from src.features.build_features import CustomStdScaler
from tests.fake_data import generate_fake_matrix


@pytest.fixture(scope="package")
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target: str
) -> FeatureParams:
    fp = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=None,
        target=target
    )
    return fp


def test_extract_target(input_data_path: str, feature_params: FeatureParams):
    data = read_data(input_data_path)
    target = extract_target(data, feature_params)
    assert isinstance(target, pd.Series)
    assert len(target) == len(data)
    assert data[feature_params.target].equals(target)


def test_custom_transformer():
    arr = generate_fake_matrix(num_rows=5, num_cols=10)
    expected_arr = (arr - arr.mean(axis=0)) / arr.std(axis=0)
    scaler = CustomStdScaler()
    scaler.fit(arr)
    transformed_arr = scaler.transform(arr)
    assert arr.shape == transformed_arr.shape
    assert np.allclose(expected_arr, transformed_arr)


def test_column_transformer(fake_pd_dataframe: pd.DataFrame, feature_params: FeatureParams):
    transformer = build_transformer(feature_params)
    transformer.fit(fake_pd_dataframe)
    sklearn.utils.validation.check_is_fitted(transformer)
    transformed_fake_pd_dataframe = make_features(transformer, fake_pd_dataframe)
    expected_rows = fake_pd_dataframe.shape[0]
    expected_cols = 30
    assert not pd.isnull(transformed_fake_pd_dataframe).any().any()
    assert isinstance(transformed_fake_pd_dataframe, pd.DataFrame)
    assert (expected_rows, expected_cols,) == transformed_fake_pd_dataframe.shape
