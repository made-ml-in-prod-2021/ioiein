from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATASET_RAW_DATA_FILE_NAME, MODELS_DIR, \
    MODEL_FILE_NAME, DATA_VOLUME_DIR, DATA_PREDICTION_DIR

with DAG(
        "predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(0, 2),
) as dag:
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=5,
        retries=5,
        filepath='/'.join(["data/raw/{{ ds }}", DATASET_RAW_DATA_FILE_NAME])
    )

    wait_for_model = FileSensor(
        task_id='wait-for-model',
        poke_interval=5,
        retries=5,
        filepath='/'.join(["data/models/{{ ds }}", MODEL_FILE_NAME])
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {DATA_RAW_DIR} --input-model-dir {MODELS_DIR} --output-dir {DATA_PREDICTION_DIR}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    #[wait_for_data, wait_for_model] >> predict
