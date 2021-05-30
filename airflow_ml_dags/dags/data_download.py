import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
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