from dataclasses import dataclass
from feature_params import FeatureParams
from split_params import SplittingParams
from train_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    logger_config: str
    split_params: SplittingParams
    feature_params: FeatureParams
    classifier_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
