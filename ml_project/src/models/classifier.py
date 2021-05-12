import numpy as np
import sklearn
import sklearn.ensemble
import pickle

from ..entities import ClassifierParams


class Classifier:
    def __init__(self, params: ClassifierParams):
        if params.model_type == "Logistic Regression":
            self.model = sklearn.linear_model.LogisticRegression(
                C=params.C, penalty=params.penalty, random_state=params.random_state, n_jobs=params.n_jobs,
                max_iter=params.max_iter
            )
        if params.model_type == "Random Forest Classifier":
            self.model = sklearn.ensemble.RandomForestClassifier(
                n_estimators=params.n_estimators, max_depth=params.max_depth, n_jobs=params.n_jobs,
                random_state=params.random_state
            )

    def fit(self, x: np.array, y: np.array) -> None:
        self.model.fit(x, y)

    def predict(self, x: np.array) -> np.array:
        return self.model.predict(x)

    def dump(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


def get_score(y_true: np.array, y_score: np.array) -> float:
    return sklearn.metrics.roc_auc_score(y_true, y_score)
