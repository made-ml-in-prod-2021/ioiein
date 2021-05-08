import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.compose
import sklearn.impute

from src.entities import FeatureParams


class CustomStdScaler(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mean) / self.std
        return x


def build_categorical_pipeline() -> sklearn.pipeline.Pipeline:
    categorical_pipeline = sklearn.pipeline.Pipeline(
        [
            ("impute", sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", sklearn.preprocessing.OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> sklearn.pipeline.Pipeline:
    num_pipeline = sklearn.pipeline.Pipeline(
        [
            ("impute", sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", CustomStdScaler()),
        ]
    )
    return num_pipeline


def make_features(transformer: sklearn.compose.ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def build_transformer(params: FeatureParams) -> sklearn.compose.ColumnTransformer:
    transformer = sklearn.compose.ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target]
    return target
