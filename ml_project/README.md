ml_project\
First homework\
Данные: https://www.kaggle.com/ronitf/heart-disease-uci\

Project Organization\
├── configs\
│   ├── logging.conf.yml <- Config for logger\
│   ├── logreg_conf.yml <- Config for train/predict with logistic regression    \
│   └── rand_forest_conf.yml <- Config for train/predict with random forest classifier \
│\
├── data\
│   └── raw            <- The original, immutable data dump.\
│\
├── models             <- Trained and serialized models, model predictions, or model summaries\
│\
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),\
│                         the creator's initials, and a short `-` delimited description, e.g.\
│                         `1.0-jqp-initial-data-exploration`.\
│\
├── src                <- Source code for use in this project.\
│   ├── data           <- Scripts to download or generate data\
│   │   └── make_dataset.py\
│   │\
│   ├── features       <- Scripts to turn raw data into features for modeling\
│   │   └── build_features.py\
│   │\
│   ├── models         <- Scripts to train models and then use trained models to make\
│   │   │                 predictions\
│   │   └── classifier.py\
│   │\
│   ├── entities       <- Dataclasses for different parametres\
│   │   ├── feature_params.py\
│   │   ├── split_params.py\
│   │   ├── train_params.py\
│   │   └── train_pipeline_params.py\
│   │\
│   └── train_predict_pipeline.py <- Script with train/predict pipeline\
│\
├── tests              <- Tests for project\
├── README.md          <- The top-level README for developers using this project.\
├── requirements.txt   <- The requirements file for reproducing the analysis environment,\ e.g.
│                         generated with `pip freeze > requirements.txt`\
└── setup.py           <- makes project pip installable (pip install -e .) so src can be imported\
\
Обучение модели\
Для обучения с логистической регрессией:\
python src\train_predict_pipeline.py configs\logreg_conf.yml train\
Для предсказания(прописан файл data\raw\sample_for_predict.csv):\
python src\train_predict_pipeline.py configs\logreg_conf.yml eval\
\
Для случайного леса:\
python src\train_predict_pipeline.py configs\rand_forest_conf.yml train\
и\
python src\train_predict_pipeline.py configs\logreg_conf.yml eval\
\
Для запуска тестов:\
pytest tests\
\
Самоанализ:\
-2) +1 \
-1) +\
 0) +2\
 1) +2\
 2) +2\
 3) +2\
 4) +3\
 5) +3(при помощи faker)\
 6) +3\
 7) +3\
 8) +3\
 9) +3\
10) +3\
11) -\
12) -\
13) +1\
\
Итого: 31\