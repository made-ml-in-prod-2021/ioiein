from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATA_VOLUME_DIR, DATASET_RAW_DATA_FILE_NAME, \
    DATASET_RAW_TARGET_FILE_NAME, DATA_PROCESSED_DIR, MODELS_DIR


with DAG(
        "train",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(2),
) as dag:
    wait_for_data = FileSensor(
        task_id='wait-for-data',
        poke_interval=5,
        retries=5,
        filepath='/'.join([DATA_RAW_DIR, DATASET_RAW_DATA_FILE_NAME])
    )

    wait_for_target = FileSensor(
        task_id='wait-for-target',
        poke_interval=5,
        retries=5,
        filepath='/'.join([DATA_RAW_DIR, DATASET_RAW_TARGET_FILE_NAME])
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {DATA_RAW_DIR} --output-dir {DATA_PROCESSED_DIR} ",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=DATA_VOLUME_DIR
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {DATA_PROCESSED_DIR}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=DATA_VOLUME_DIR,
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input-dir {DATA_PROCESSED_DIR} --output-dir {MODELS_DIR}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=DATA_VOLUME_DIR,
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {DATA_PROCESSED_DIR} --input-model-dir {MODELS_DIR}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=DATA_VOLUME_DIR,
    )

    [wait_for_data, wait_for_target] >> preprocess >> split >> train >> validate
