import time
import pandas as pd
from pprint import pprint
import sklearn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import logging
import coloredlogs
from datetime import datetime
coloredlogs.install(level='INFO', fmt='%(asctime)s,%(msecs)03d:(%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)


class Engine:
    """Engine represents the orchestrator class for training, registering models and making predictions."""

    def __init__(self, encoder_path: str, mlflow_tracking_uri: str):
        self.encoder_path = encoder_path
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()

    def train(self, x_train, y_train, x_test, y_test):
        MODEL_NAME = 'KNN'  # Add your model name here. Example: clinical_ner
        EXPERIMENT_NAME = 'testing-mlflow'  # Add your experiment name here. Example: testing_dropout
        n_neighbors = 5

        EXPERIMENT_ID = mlflow.create_experiment(f"{MODEL_NAME}_{EXPERIMENT_NAME}_N={n_neighbors}_{datetime.now()}")
        with mlflow.start_run(experiment_id=EXPERIMENT_ID) as run:
            # train model
            # train a KNN model

            RUN_ID = run.info.run_id
            logger.info(f"Run id: {RUN_ID}")
            logger.info(f"Exp id: {EXPERIMENT_ID}")

            logger.info("Start training %s", MODEL_NAME)
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            start = time.time()
            knn.fit(x_train, y_train)
            end = time.time()
            ELAPSED_SEC_TRAINING = end - start
            logger.info("end training")

            # inference
            logger.info("Start inference...")
            start = time.time()
            y_pred = knn.predict(x_test)
            end = time.time()
            ELAPSED_SEC_INFERENCE = end - start

            # evaluation metrics
            recall, precision, f1, accuracy = self.evaluate(y_test, y_pred)
            logger.info("Metrics - Recall: %.2f", recall)
            logger.info("Metrics - Precision: %.2f", precision)
            logger.info("Metrics - F1-Score: %.2f", f1)
            logger.info("Metrics - Accuracy: %.2f", accuracy)

            logger.info(f"Training dataset size: {x_train.shape}")     
            logger.info(f"Training time (sec): {ELAPSED_SEC_TRAINING:.3f}")
            logger.info(f"Inference dataset size: {x_test.shape}")
            logger.info(f"Inference time (sec): {ELAPSED_SEC_INFERENCE:.3f}")

            # store with mlflow
            logger.info("Storing the model info into mlflow")
            mlflow.log_param("training_size", x_train.shape)
            mlflow.log_param("training_time", ELAPSED_SEC_TRAINING)
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("test_size", {x_test.shape})
            mlflow.log_param("test_time", ELAPSED_SEC_INFERENCE)
            mlflow.log_param("run_id", RUN_ID)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_artifact(self.encoder_path)

            PIP_REQUIREMENTS = [f"scikit-learn=={sklearn.__version__}",
                                f"mlflow=={mlflow.__version__}",
                                f"coloredlogs=={coloredlogs.__version__}"]
            artifact_path = f"{MODEL_NAME}__N={n_neighbors}_{EXPERIMENT_NAME}"
            mlflow.sklearn.log_model(knn,
                                     artifact_path=artifact_path,
                                     pip_requirements=PIP_REQUIREMENTS)
            logger.info("Done")
            registered_model_name = f"{MODEL_NAME}_N={n_neighbors}"
            return RUN_ID, artifact_path, registered_model_name

    def register_model(self, run_id: str, artifact_path: str, registered_model_name: str):
        result = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", registered_model_name)
        logger.info(f"Model: {registered_model_name}, version: {result} has been successfully registered.")

    def evaluate(self, actual, prediction):
        return recall_score(actual, prediction), precision_score(actual, prediction),\
           f1_score(actual, prediction), accuracy_score(actual, prediction)

    def predict(self, data: pd.DataFrame, model_name: str, model_version: str):
        # featurize
        data.drop("UserId", axis=1, inplace=True)

        # load encoder
        ohe = joblib.load(self.encoder_path)
        X = ohe.transform(data).toarray()

        try:
            model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
        except Exception as e:
            logger.exception("Unable to load model, please check your model info: %s", e)
            raise e

        y_proba = model.predict_proba(X)
        # return prediction probability for positive class (Fake)
        return y_proba[:, 1]

    def get_models(self):
        """Display list of all registered models."""
        from pprint import pprint
        for registered_models in self.client.list_registered_models():
            pprint(dict(registered_models), indent=4)

    def get_model_versions(self, model_name: str):
        """Display list of all registered models version available,
        for a given `model_name`."""
        client = MlflowClient()
        for versions in client.search_model_versions(f"name='{model_name}'"):
            pprint(dict(versions), indent=4)
