import logging
import pandas as pd
import sys
from pathlib import Path
import joblib
import coloredlogs
from typing import Any
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import mlflow
import mlflow.sklearn

import src.config as config
from urllib.parse import urlparse
coloredlogs.install(level='INFO', fmt='%(asctime)s,%(msecs)03d:(%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)


def prepare_train(ohe: OneHotEncoder, path_to_csv: str = "../data/fake_users.csv",
                  undersampling: bool = True) -> Any:
    try:
        data = pd.read_csv(path_to_csv)
    except Exception as e:
        logger.exception("Unable to load trainingCSV, check the file path. Error: %s", e)

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.drop("UserId", axis=1, inplace=True)

    if undersampling:
        original_data = data.copy()
        new_data_feat = original_data.sample(frac=1)

        # determine cardinality of minority class (Fake = 1)
        nb_fake = new_data_feat['Fake'].value_counts()[1]
        
        # constitute fake and not fake dataframes
        fake_df = new_data_feat.loc[new_data_feat['Fake'] == 1]
        not_fake_df = new_data_feat.loc[new_data_feat['Fake'] == 0][:nb_fake]

        # concatenate  fake and not fake dataframes
        normal_distributed_df = pd.concat([fake_df, not_fake_df], ignore_index=True)
        new_df = normal_distributed_df.sample(frac=1, random_state=42, ignore_index=True)

        return split_and_featurize(ohe, new_df)
    else:
        return split_and_featurize(ohe, data)


def split_and_featurize(ohe: OneHotEncoder, dataframe: pd.DataFrame) -> Any:
    # split X, y
    y = dataframe.pop("Fake")
    X = dataframe

    # apply featurization
    ohe.fit(X)
    X = ohe.transform(X).toarray()
    return X, y


def prepare_test(ohe: OneHotEncoder, path_to_csv: str = "../data/fake_users_test.csv") -> Any:
    try:
        data_test = pd.read_csv(path_to_csv)
    except Exception as e:
        logger.exception("Unable to load trainingCSV, check the file path. Error: %s", e)
    
    # split X, y data
    y_test = data_test.pop("Fake")
    X_test = data_test

    # featurize
    X_test_feat = X_test.iloc[:, 1:]
    X_test_feat = ohe.transform(X_test_feat).toarray()
    return X_test_feat, y_test


def evaluation_metrics(actual, prediction):
    return recall_score(actual, prediction), precision_score(actual, prediction),\
           f1_score(actual, prediction), accuracy_score(actual, prediction)


def save_encoder(ohe: OneHotEncoder, output: str):
    path = Path(output)
    # create parent folder if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(ohe, output)
    except FileNotFoundError as e:
        logger.exception("Invalid destination for encoder: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    with mlflow.start_run(run_name="KNN_TRAINING_RUN"):
        undersampling = False
        logger.info("start model training...")

        logger.info("Prepare train/test data")
        # prepare dataset
        ohe = OneHotEncoder(handle_unknown='ignore')
        X_train, y_train = prepare_train(ohe, config.train_dataset_path, undersampling)
        X_test, y_test = prepare_test(ohe, config.test_dataset_path)
        save_encoder(ohe, config.ohe_features_path)

        # train a KNN model
        logger.info("Training KNN")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # evaluation metrics
        recall, precision, f1, accuracy = evaluation_metrics(y_test, y_pred)
        logger.info("Metrics - Recall: %.2f", recall)
        logger.info("Metrics - Precision: %.2f", precision)
        logger.info("Metrics - F1-Score: %.2f", f1)
        logger.info("Metrics - Accuracy: %.2f", accuracy)

        # store model using mlflow
        logger.info("Storing the model using mlflow")
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)

        run_id = mlflow.active_run().info.run_id
        artifact_path = "model"
        model_name = "knn_n_5_for_undersampled_dataset"

        model_uri = f"runs:/{run_id}/{artifact_path}"

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info("model stored at %s",  mlflow.get_artifact_uri())
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(knn, "knn-model", registered_model_name="knn-model-n=5")
        else:
            mlflow.sklearn.log_model(knn, "model")
