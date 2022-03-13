from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import logging
import coloredlogs
import joblib
import pandas as pd
coloredlogs.install(level='INFO', fmt='%(asctime)s,%(msecs)03d:(%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self, encoder_path):
        self.encoder_path = encoder_path

    def save_encoder(self, ohe: OneHotEncoder):
        """Save encoder used for featurization."""
        path = Path(self.encoder_path)
        # create parent folder if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(ohe, self.encoder_path)
        except FileNotFoundError as e:
            logger.exception("Invalid destination for encoder: %s", e)

    def preprocess(self, data: pd.DataFrame, y_label: str = "Fake"):
        """Separate data from class label, and remove UsersId."""
        y = data.pop(y_label)
        # remove the id label
        x = data.iloc[:, 1:]

        return x, y

    def clean(self, data: pd.DataFrame):
        """Clean data by removing the Unnamed field."""
        print(data.head())
        data.drop("Unnamed: 0", axis=1, inplace=True)
        return data

    def load_data(self, train_dataset_path: str, test_dataset_path: str):
        """Load .csv file into dataframe."""
        try:
            data_train = pd.read_csv(train_dataset_path)
        except Exception as e:
            logger.exception("Unable to load trainCSV, check the file path. Error: %s", e)
            return
        try:
            data_test = pd.read_csv(test_dataset_path)
        except Exception as e:
            logger.exception("Unable to load testCSV, check the file path. Error: %s", e)
            return

        return data_train, data_test

    def create(self, train_dataset_path: str, test_dataset_path: str, sampling: bool):
        """Create proper featurized train and test set for training."""
        ohe = OneHotEncoder(handle_unknown='ignore')
        self.save_encoder(ohe)

        try:
            data_train, data_test = self.load_data(train_dataset_path, test_dataset_path)
        except FileNotFoundError:
            return
        except TypeError:
            return
        except Exception as e:
            raise Exception("unable to prepare data %s", e)

        data_train = self.clean(data_train)

        x_train, y_train = self.preprocess(data_train)
        x_test, y_test = self.preprocess(data_test)

        # featurize
        ohe.fit(x_train)
        x_train = ohe.transform(x_train).toarray()
        x_test = ohe.transform(x_test).toarray()
        self.save_encoder(ohe)

        return x_train, y_train, y_train,  x_test, y_test
