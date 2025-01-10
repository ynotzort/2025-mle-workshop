import os
os.environ.setdefault("MODEL_PATH", "./models/2022-01.bin")

from duration_prediction_serve.serve import prepare_features


def test_prepare_features():
    ride = {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 20,
    }

    expected = {
        "PULocationID": "100",
        "DOLocationID": "102",
        "trip_distance": 20,
    }
    result = prepare_features(ride)
    assert result == expected
