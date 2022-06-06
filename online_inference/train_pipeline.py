import json
import logging
import sys
import click

from src.data.make_dataset import read_data, split_train_val_data

from src.features.build_features import (
    process_features,
    build_transformer,
    extract_target,
    extract_features,
    serialize_transformer,
)
from src.models.fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model,
)
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)


logger = logging.getLogger(__name__)


def setup_logging():
    """Настройка логгера"""
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(stdout_handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"Старт пайплайна с параметрами:\n{training_pipeline_params}")

    logger.info("Чтение данных...")
    data = read_data(training_pipeline_params.input_data_path)

    logger.info("Сплит данных...")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    logger.info("Препроцессинг...")
    train_features = extract_features(train_df, training_pipeline_params.feature_params)
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_features)
    serialize_transformer(transformer, training_pipeline_params.output_transformer_path)

    train_features = process_features(transformer, train_features)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    val_features = extract_features(val_df, training_pipeline_params.feature_params)
    val_features = process_features(transformer, val_features)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    logger.info("Обучение модели...")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    logger.info("Подсчет метрик...")
    preds = predict_model(model, val_features)
    metrics = evaluate_model(preds, val_target)
    with open(training_pipeline_params.metric_path, mode="w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Значения метрик: {metrics}")

    logger.info("Сериализация модели...")
    serialize_model(model, training_pipeline_params.output_model_path)


@click.command()
@click.option("--cfg_path", "-cfg", help="config path")
def train_pipeline_from_cfg(cfg_path):
    setup_logging()
    params = read_training_pipeline_params(cfg_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_from_cfg()
