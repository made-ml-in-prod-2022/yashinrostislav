from dataclasses import dataclass


@dataclass
class SplittingParams:
    train_size: float
    random_state: int
