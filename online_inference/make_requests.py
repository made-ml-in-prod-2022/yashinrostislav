import requests
import logging
import sys
import pandas as pd
from app import CONFIG_PATH

from src.entities.online_inference_app_params import read_online_inference_app_params

DATA_PATH = "data/raw/sample_for_pred.csv"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def send_requests():
    data = pd.read_csv(DATA_PATH)
    online_inference_app_params = read_online_inference_app_params(CONFIG_PATH)
    url = f"http://{online_inference_app_params.host}:{online_inference_app_params.port}/predict/"
    for i in range(len(data)):
        json_data = {
            "data": [data.iloc[i].values.tolist()],
            "features": list(data.columns),
        }
        response = requests.post(url, json=json_data)
        logger.info(
            f"Request: {json_data}, response_code: {response.status_code}, response_json: {response.json()}"
        )


if __name__ == "__main__":
    send_requests()
