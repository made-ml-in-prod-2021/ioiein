from typing import List, Tuple

import pytest
import pandas as pd
import sklearn.ensemble
import sklearn.utils.validation

from src.entities import FeatureParams, ClassifierParams
from src.data import read_data
from src.features import build_transformer, make_features, extract_target
from src.models import Classifier


@pytest.fixture(scope="package")
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target: str,
) -> FeatureParams:
    fp = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=None,
        target=target
    )
    return fp


@pytest.fixture(scope="package")
def classifier_params() -> ClassifierParams:
    cp = ClassifierParams(
        model_type="Random Forest Classifier",
        n_estimators=250,
        max_depth=6,
        n_jobs=-1,
        random_state=225,
        C=None,
        max_iter=None,
        penalty=None
    )
    return cp


@pytest.fixture(scope="package")
def preprocess_data(
    input_data_path: str, feature_params: FeatureParams
) -> Tuple[pd.Series, pd.DataFrame]:
    data = read_data(input_data_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    transformed_features = make_features(transformer, data)
    target = extract_target(data, feature_params)
    return target, transformed_features


def test_train_model(
    classifier_params: ClassifierParams,
    preprocess_data: Tuple[pd.Series, pd.DataFrame],
):
    target, transformed_features = preprocess_data
    model = Classifier(classifier_params)
    model.fit(transformed_features, target)
    sklearn.utils.validation.check_is_fitted(model.model)
    assert isinstance(model.model, sklearn.ensemble.RandomForestClassifier)
