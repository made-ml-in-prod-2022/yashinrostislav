from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "yashinrostislav",
    "email": ["yashinrostislav@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}
MODEL_DIR = "/data/model/{{ ds }}"
RAW_DATA_DIR = "/data/raw/{{ ds }}"
PREPROCESSED_DATA_DIR = "/data/preprocessed/{{ ds }}"
VOLUME = "/home/rostislav/vcs/made-ml-in-prod-2022/yashinrostislav/airflow_ml_dags/data:/data"

