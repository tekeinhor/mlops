import os
from src.config import settings
from src.engine import Engine
import numpy as np
import pytest
import mlflow
CurrentWorkingDir = os.getcwd()
test_model_path = "testdata/mlruns/0/fd6986f2b8104c81bad25bceb61f3492/artifacts/model"


engine = Engine(settings.ENCODER_PATH, settings.MLFLOW.TRACKING_URI)


@pytest.mark.parametrize(
    "X_list, model_uri, expected_y",
    [
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
         f"{CurrentWorkingDir}/{test_model_path}", [0.2, 0.1]),
    ],
    ids=[
        "normal",
    ],
)
def test_predict(X_list, model_uri, expected_y):
    X = np.array(X_list)
    actual_y = engine.predict(X, model_uri)
    assert len(expected_y) == len(actual_y)


@pytest.mark.parametrize(
    "X_list, model_uri, error",
    [
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            "",  mlflow.exceptions.MlflowException),
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            "toto", OSError),
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
            "toto", OSError),
    ],
    ids=[
        "empty model path",
        "invalid model path",
        "incorrect feature size",
    ],
)
def test_predict_error_cases(X_list, model_uri, error):
    X = np.array(X_list)
    with pytest.raises(error):
        _ = engine.predict(X, model_uri)
