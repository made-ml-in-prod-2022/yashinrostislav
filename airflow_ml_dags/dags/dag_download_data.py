from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.operators.dummy import DummyOperator

from const import DEFAULT_ARGS, RAW_DATA_DIR, VOLUME

with DAG(
    "dag_download_data",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(14),
) as dag:
    start_download = DummyOperator(task_id="start-download-data")

    download_data = DockerOperator(
        image="airflow-download-data",
        command=f"--output-dir {RAW_DATA_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download-data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    start_download >> download_data
