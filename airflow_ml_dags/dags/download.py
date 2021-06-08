from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATA_VOLUME_DIR, DATASET_RAW_DATA_FILE_NAME, DATA_PROCESSED_DIR


with DAG(
    "download",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(0, 2),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command=f"--output-dir {DATA_RAW_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )
