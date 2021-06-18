import sys

import pytest
from airflow.models import DagBag

sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


def test_data_download_dag(dag_bag):
    dag = dag_bag.dags['download']

    dag_flow = {
        'docker-airflow-download': [],
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])


def test_train_dag(dag_bag):
    dag = dag_bag.dags['train']

    dag_flow = {
        'docker-airflow-preprocess': ['docker-airflow-split'],
        'docker-airflow-split': ['docker-airflow-train'],
        'docker-airflow-train': ['docker-airflow-validate'],
        'docker-airflow-validate': []
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])


def test_predict_dag(dag_bag):
    dag = dag_bag.dags['predict']

    dag_flow = {
        'docker-airflow-predict': []
    }

    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])
