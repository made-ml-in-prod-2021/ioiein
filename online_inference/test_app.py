import json

import pytest
from fastapi.testclient import TestClient

from app import app, load_model
from src.entities import HeartData


@pytest.fixture(scope="session", autouse=True)
def initialize_model():
    load_model()


@pytest.fixture()
def test_data():
    data = [HeartData(id=0, age=18, sex=0, cp=0, trestbps=100, chol=245, fbs=0, restecg=0, thalach=100,
                      exang=0, oldpeak=0, slope=0, ca=0, thal=0)]
    return data


def test_main_endpoint_works_correctly():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code


def test_status_endpoint_works_correctly():
    with TestClient(app) as client:
        expected_response = "Model is ready"
        response = client.get("/status")
        assert 200 == response.status_code
        assert expected_response == response.json()


def test_predict_endpoint_works_correctly(test_data):
    with TestClient(app) as client:
        response = client.post(
            "/predict", data=json.dumps([x.__dict__ for x in test_data])
        )
        assert 200 == response.status_code
        assert len(response.json()) == len(test_data)
        assert 0 == response.json()[0]["id"]
        assert response.json()[0]["target"] in [0, 1]