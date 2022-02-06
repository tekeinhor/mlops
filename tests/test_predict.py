from src.predict import make_prediction
import numpy as np
import os
import pytest
CurrentWorkingDir = os.getcwd()
test_model_path = "testdata/mlruns/0/fd6986f2b8104c81bad25bceb61f3492/artifacts/model"


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
def test_make_prediction(X_list, model_uri, expected_y):
    X = np.array(X_list)
    actual_y = make_prediction(X, model_uri)
    assert len(expected_y) == len(actual_y)


@pytest.mark.parametrize(
    "X_list, model_uri, error",
    [
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            "", ValueError),
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            "toto", SystemExit),
        ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
            "toto", SystemExit),
    ],
    ids=[
        "empty model path",
        "invalid model path",
        "incorrect feature size",
    ],
)
def test_make_prediction_error_cases(X_list, model_uri, error):
    X = np.array(X_list)
    with pytest.raises(error):
        _ = make_prediction(X, model_uri)
