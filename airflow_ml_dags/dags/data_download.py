from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATA_VOLUME_DIR


with DAG(
    "data_download",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(0, 2),
) as dag:
    data_download = DockerOperator(
        image="airflow-download",
        command=f"--output_dir {DATA_RAW_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )
