from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelNameConfig:
    """
    Model Config using dataclass
    """
    model_name: str = "LinearRegression"
    model_kwargs: Dict = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
