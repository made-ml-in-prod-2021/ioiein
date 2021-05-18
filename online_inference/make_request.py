import json
import requests

import pandas as pd

DATA_PATH = "sample_for_predict.csv"


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    data["id"] = range(len(data))
    request_data = data.to_dict(orient="records")
    print("Sample of request data:")
    print(request_data[0:3])
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json.dumps(request_data)
    )
    print(f"Response status code: {response.status_code}")
    print("Sample of response data:")
    print(response.json())
