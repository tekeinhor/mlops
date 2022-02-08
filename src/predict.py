#!/usr/bin/env python
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
import sys
import coloredlogs
coloredlogs.install(level='INFO', fmt='%(asctime)s,%(msecs)03d:(%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)


def make_prediction(X: pd.DataFrame, model_uri: str):
    if model_uri == "":
        logger.exception("Unable to load model, empty model_uri")
        raise ValueError("empty model_uri")
    logger.info("search model at location %s", model_uri)
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.exception("Unable to load model, check the file path. Error: %s", e)
        sys.exit(1)
    y_proba = model.predict_proba(X)
    # return prediction probability for positive class (Fake)
    return y_proba[:, 1]
