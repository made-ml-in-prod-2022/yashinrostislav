import contextlib
import time
import threading
from typing import List, Union
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

from src.entities.online_inference_app_params import read_online_inference_app_params
from src.features.build_features import process_features
from src.models.fit_predict import (
    predict_model,
    load_model,
    load_transformer,
)

CONFIG_PATH = "configs/online_inference_app_config.yaml"
APP_START_DELAY = 20
APP_LIFE_DURATION = 120

online_inference_app_params = read_online_inference_app_params(CONFIG_PATH)
model = load_model(online_inference_app_params.model_path)
transformer = load_transformer(online_inference_app_params.transformer_path)
model_features = transformer._features

app = FastAPI()


class Classifier(BaseModel):
    data: List[List[Union[float, int]]]
    features: List[str]

    @validator("features")
    def validate_features(cls, features):
        if set(features) != set(model_features):
            raise HTTPException(
                status_code=400,
                detail=f"Список фичей не совпадает со списком обученной модели. Ожидается: {model_features}"
            )
        return features

    @validator("data")
    def validate_data(cls, data):
        _, num_cols = pd.DataFrame(data).shape
        if num_cols != len(model_features):
            raise HTTPException(
                status_code=400,
                detail=f"Неправильное количество столбцов в датасете ({num_cols}). Ожидается: {len(model_features)}"
            )
        return data


class ModelResponse(BaseModel):
    label: int


def predict(data: List, features: List[str]) -> List[ModelResponse]:
    features = pd.DataFrame(data, columns=features)
    features = process_features(transformer, features)
    preds = predict_model(model, features)
    res = [ModelResponse(label=label) for label in preds]
    return res


@app.get("/")
async def app_root():
    return "Привет!"


@app.post("/predict", response_model=List[ModelResponse])
async def app_predict(request: Classifier):
    return predict(request.data, request.features)


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def app_run():
    config = uvicorn.Config(app, host=online_inference_app_params.host, port=online_inference_app_params.port)
    server = Server(config=config)

    time.sleep(APP_START_DELAY)
    with server.run_in_thread():
        time.sleep(APP_LIFE_DURATION)


if __name__ == "__main__":
    app_run()