from typing import List

import pytest
import pandas as pd
from faker import Faker


FAKE_DATA_SIZE = 150
FAKE_DATA_PREDICT_SIZE = 10


@pytest.fixture(scope="session")
def input_data_path() -> str:
    return "tests/fake_data.csv"


@pytest.fixture(scope="session")
def input_data_predict_path() -> str:
    return "tests/fake_data_for_predict.csv"


@pytest.fixture(scope="session")
def output_data_predict_path() -> str:
    return "tests/test_predictions.csv"


@pytest.fixture(scope="session")
def output_model_path() -> str:
    return "tests/test_model.pkl"


@pytest.fixture(scope="session")
def transformer_path() -> str:
    return "tests/test_transformer.pkl"


@pytest.fixture(scope="session")
def target() -> str:
    return "target"


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="session")
def fake_pd_dataframe() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(25)
    data = {
        "age": [fake.pyint(min_value=25, max_value=80) for _ in range(FAKE_DATA_SIZE)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_SIZE)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(FAKE_DATA_SIZE)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(FAKE_DATA_SIZE)],
        "chol": [fake.pyint(min_value=100, max_value=600) for _ in range(FAKE_DATA_SIZE)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_SIZE)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(FAKE_DATA_SIZE)],
        "thalach": [fake.pyint(min_value=70, max_value=205) for _ in range(FAKE_DATA_SIZE)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_SIZE)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(FAKE_DATA_SIZE)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(FAKE_DATA_SIZE)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(FAKE_DATA_SIZE)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(FAKE_DATA_SIZE)],
        "target": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_SIZE)],
    }
    return pd.DataFrame(data=data)


@pytest.fixture(scope="session")
def fake_pd_dataframe_predict() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(225)
    data = {
        "age": [fake.pyint(min_value=25, max_value=80) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "chol": [fake.pyint(min_value=100, max_value=600) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "thalach": [fake.pyint(min_value=70, max_value=205) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(FAKE_DATA_PREDICT_SIZE)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(FAKE_DATA_PREDICT_SIZE)],
    }
    return pd.DataFrame(data=data)