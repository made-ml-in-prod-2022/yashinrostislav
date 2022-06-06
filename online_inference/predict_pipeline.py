import logging
import sys
import click

from src.data.make_dataset import read_data

from src.features.build_features import process_features
from src.models.fit_predict import (
    predict_model,
    load_model,
    load_transformer,
    save_preds,
)
from src.entities.predict_pipeline_params import (
    PredictingPipelineParams,
    read_predicting_pipeline_params,
)

CONFIG_PATH = "configs/train_config_lr.yaml"

logger = logging.getLogger(__name__)


def setup_logging():
    """Настройка логгера"""
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(stdout_handler)


def predict_pipeline(predicting_pipeline_params: PredictingPipelineParams) -> None:
    logger.info(f"Старт пайплайна с параметрами:\n{predicting_pipeline_params}")

    logger.info("Чтение данных...")
    features = read_data(predicting_pipeline_params.input_data_path)

    logger.info("Загрузка трансформера...")
    transformer = load_transformer(predicting_pipeline_params.transformer_path)

    logger.info("Применение трансформера...")
    features = process_features(transformer, features)

    logger.info("Загрузка модели...")
    model = load_model(predicting_pipeline_params.model_path)

    logger.info("Применение модели...")
    preds = predict_model(model, features)

    logger.info("Сохранение предсказаний...")
    save_preds(preds, predicting_pipeline_params.output_preds_path)


@click.command()
@click.option("--cfg_path", "-cfg", help="config path")
def predict_pipeline_from_cfg(cfg_path):
    setup_logging()
    params = read_predicting_pipeline_params(cfg_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_from_cfg()
