from typing import List

import pytest
import pandas as pd

from src.entities import TrainingPipelineParams, SplittingParams, FeatureParams, ClassifierParams
from src.train_predict_pipeline import train_pipeline, predict_pipeline


@pytest.fixture(scope="package")
def train_pipeline_params(
    fake_pd_dataframe: pd.DataFrame,
    fake_pd_dataframe_predict: pd.DataFrame,
    input_data_path: str,
    input_data_predict_path: str,
    output_data_predict_path: str,
    output_model_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target: str,
    transformer_path: str,
) -> TrainingPipelineParams:
    fake_pd_dataframe.to_csv(input_data_path, index=False)
    fake_pd_dataframe_predict.to_csv(input_data_predict_path, index=False)
    tpp = TrainingPipelineParams(
        input_data_path=input_data_path,
        input_data_predict_path=input_data_predict_path,
        output_data_predict_path=output_data_predict_path,
        output_model_path=output_model_path,
        output_transformer_path=transformer_path,
        split_params=SplittingParams(validation_size=0.2, random_state=25),
        feature_params=FeatureParams(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            features_to_drop=None,
            target=target,
        ),
        classifier_params=ClassifierParams(
            model_type="Logistic Regression",
            C=1.0,
            penalty="l2",
            random_state=25,
            n_jobs=-1,
            max_iter=300,
            max_depth=None,
            n_estimators=None,
        ),
        logger_config="configs/logging.conf.yml"
    )
    return tpp


def test_full_pipeline(train_pipeline_params: TrainingPipelineParams):
    score = train_pipeline(train_pipeline_params)
    assert 0 < score <= 1


def test_predict_pipeline(train_pipeline_params: TrainingPipelineParams):
    predict = predict_pipeline(train_pipeline_params)
    assert len(predict) == 10