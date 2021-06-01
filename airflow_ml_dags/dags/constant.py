import datetime
from airflow.models import Variable


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

DATA_RAW_DIR = "/data/raw/{{ ds }}"
DATA_VOLUME_DIR = Variable.get("data_path")

DATASET_RAW_DATA_FILE_NAME = "data.csv"
DATASET_RAW_TARGET_FILE_NAME = "target.csv"

DATA_PROCESSED_DIR = "/data/processed/{{ ds }}"

MODELS_DIR = "/data/models/{{ ds }}"
MODEL_FILE_NAME = "model"

DATA_PREDICTION_DIR = "/data/predictions/{{ ds }}"