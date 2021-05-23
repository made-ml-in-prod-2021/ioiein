import logging
import os
import pickle
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline
import yaml

from src.entities import HeartResponse, HeartData
from src.validate import validate_data

APPLICATION_NAME = "homework_2"

DEFAULT_PATH_LOG_CONFIG = "logging.conf.yml"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging(path: str) -> None:
    with open(path) as config_f:
        logging.config.dictConfig(yaml.safe_load(config_f))


def load_object(path_to_model: str) -> Pipeline:
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    logger.info("model loaded")
    return model


pipeline: Optional[Pipeline] = None


def make_prediction(data: List[HeartData]) -> List[HeartResponse]:
    data = pd.DataFrame(x.__dict__ for x in data)
    ids = [int(x) for x in data.id]
    predicts = pipeline.predict(data.drop("id", axis=1))

    return [
        HeartResponse(id=id_, target=int(target_))
        for id_, target_ in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    logger.info("service started")
    model_path = os.getenv("PATH_TO_MODEL", "model.pkl")
    logger.info("model path got")
    if model_path is None:
        e = f"PATH_TO_MODEL {model_path} is None"
        logger.error(e)
        raise FileExistsError(e)
    global pipeline
    pipeline = load_object(model_path)


@app.get("/status")
def status() -> str:
    if pipeline is not None:
        return f"Model is ready"
    else:
        return f"Model is not ready"


@app.api_route("/predict", response_model=List[HeartResponse], methods=["GET", "POST"])
def predict(request: List[HeartData]):
    for data in request:
        is_valid = validate_data(data)
        if not is_valid:
            raise HTTPException(status_code=400)
    return make_prediction(request)


if __name__ == "__main__":
    setup_logging(DEFAULT_PATH_LOG_CONFIG)
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
