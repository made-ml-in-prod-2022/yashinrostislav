from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingParams:
    model_type: str
    model_params: Dict[str, Any]
