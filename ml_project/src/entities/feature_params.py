from dataclasses import dataclass
from typing import Optional, List


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: Optional[List[str]]
    target: Optional[str]
