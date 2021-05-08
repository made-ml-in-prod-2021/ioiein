import logging
import logging.config
import yaml
import click
import pickle
import pandas as pd

from src.data import read_data, split_train_val_data
from src.entities import TrainingPipelineParams, read_training_pipeline_params
from src.models import Classifier, get_score
from src.features import make_features, extract_target, build_transformer


APPLICATION_NAME = "homework_1"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging(path: str) -> None:
    with open(path) as config_f:
        logging.config.dictConfig(yaml.safe_load(config_f))


def train_pipeline(params: TrainingPipelineParams) -> float:
    logger.info(f"train pipeline started")
    data = read_data(params.input_data_path)
    logger.info(f"data opened, shape {data.shape}")
    train_data, val_data = split_train_val_data(data, params.split_params)
    logger.info(f"data split")
    logger.debug(f"train shape {train_data.shape}")
    logger.debug(f"validation shape {val_data.shape}")
    transformer = build_transformer(params.feature_params)
    logger.info(f"transformer initiated")
    transformer.fit(train_data.drop(columns=['target']))
    logger.info(f"transformer fitted")
    train_features = make_features(transformer, train_data.drop(columns=['target']))
    train_target = extract_target(train_data, params.feature_params)
    logger.info(f"created train features and target")
    model = Classifier(params.classifier_params)
    logger.info(f"model created")
    model.fit(train_features, train_target)
    logger.info(f"model fitted")
    val_features = make_features(transformer, val_data.drop(columns=['target']))
    val_target = extract_target(val_data, params.feature_params)
    logger.info(f"created validation features and target")
    predicts = model.predict(val_features)
    logger.info(f"made predicts")
    score = get_score(val_target, predicts)
    logger.debug(f"roc-auc score {score}")
    model.dump(params.output_model_path)
    logger.info(f"model dumped")
    with open(params.output_transformer_path, "wb") as f:
        pickle.dump(transformer, f)
    logger.info(f"transformer dumped")
    logger.info(f"train pipeline finished")
    return score


def predict_pipeline(params: TrainingPipelineParams) -> pd.DataFrame:
    logger.info(f"predict pipeline started")
    data = read_data(params.input_data_predict_path)
    logger.info(f"data opened")
    logger.debug(f"data shape {data.shape}")
    with open(params.output_model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"model loaded")
    with open(params.output_transformer_path, "rb") as f:
        transformer = pickle.load(f)
    logger.info(f"transformer loaded")
    transformed_data = make_features(transformer, data)
    predicts = model.predict(transformed_data)
    logger.info(f"predicts ready")
    pd.DataFrame(predicts, columns=["target"]).to_csv(params.output_data_predict_path, index=False)
    logger.info(f"predict pipeline finished")
    return pd.DataFrame(predicts, columns=["target"])


@click.command(name="train_predict_pipeline")
@click.argument("config_path")
@click.argument("train_eval")
def pipeline_command(config_path: str, train_eval: str):
    params = read_training_pipeline_params(config_path)
    setup_logging(params.logger_config)
    if train_eval == "train":
        train_pipeline(params)
    elif train_eval == "eval":
        predict_pipeline(params)


if __name__ == "__main__":
    pipeline_command()
