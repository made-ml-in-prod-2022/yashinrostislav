from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class PredictingPipelineParams:
    input_data_path: str
    model_path: str
    transformer_path: str
    output_preds_path: str


PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)


def read_predicting_pipeline_params(path: str) -> PredictingPipelineParams:
    """Чтение параметров пайплайна"""
    with open(path, mode="r") as f:
        config_dict = yaml.safe_load(f)
        schema = PredictingPipelineParamsSchema().load(config_dict)
        return schema



