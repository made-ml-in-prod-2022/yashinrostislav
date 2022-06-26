from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from const import DEFAULT_ARGS, RAW_DATA_DIR, PREPROCESSED_DATA_DIR, MODEL_DIR, VOLUME

with DAG(
    "dag_train_model",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    start_date=days_ago(14),
) as dag:
    start_train_model = DummyOperator(task_id="start-train-model")

    preprocess_data = DockerOperator(
        image="airflow-preprocess-data",
        command=f"--input-dir {RAW_DATA_DIR} --output-dir {PREPROCESSED_DATA_DIR}",
        task_id="docker-airflow-preprocess-data",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    split_data = DockerOperator(
        image="airflow-split-data",
        command=f"--input-dir {PREPROCESSED_DATA_DIR}",
        task_id="docker-airflow-split-data",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    train_model = DockerOperator(
        image="airflow-train-model",
        command=f"--input-dir {PREPROCESSED_DATA_DIR} --model-path {MODEL_DIR}",
        task_id="docker-airflow-train-model",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    validate_model = DockerOperator(
        image="airflow-validate-model",
        command=f"--model-dir {MODEL_DIR} --data-path {PREPROCESSED_DATA_DIR}",
        task_id="docker-airflow-validate-model",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    start_train_model >> preprocess_data >> split_data >> train_model >> validate_model
