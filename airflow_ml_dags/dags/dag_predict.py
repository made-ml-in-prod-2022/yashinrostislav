from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from const import DEFAULT_ARGS, MODEL_DIR, PREPROCESSED_DATA_DIR, VOLUME


with DAG(
    "dag_predict",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(14),
) as dag:

    start_predict = DummyOperator(task_id="start-predict")

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--model-dir {MODEL_DIR} --data-dir {PREPROCESSED_DATA_DIR}",
        task_id="docker-airflow-predict",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    start_predict >> predict
