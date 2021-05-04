from dataclasses import dataclass
from typing import Optional


@dataclass()
class TrainingParams:
    model_type: str
    penalty: Optional[str]
    C: Optional[float]
    max_iter: Optional[int]
    n_estimators: Optional[int]
    max_depth: Optional[int]
    n_jobs: int = -1
    random_state: int = 25
