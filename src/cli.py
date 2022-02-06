import argparse
import sys
import pandas as pd
import logging
import coloredlogs
from src.predict import make_prediction
import src.config as config
import joblib
from pathlib import Path
import os
coloredlogs.install(level='INFO', fmt='%(asctime)s,%(msecs)03d:(%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)


def get_latest_model_dir(dirpath: str) -> Path:
    print("toto")
    p = Path(dirpath)
    print("tata")
    print(p.iterdir())
    try:
        paths = [x for x in p.iterdir() if x.is_dir()]
    except Exception as e:
        logger.exception("unable to get latest model: are you sure a model has been trained: %s", e)
        sys.exit(1)

    paths = sorted(paths, key=os.path.getmtime)
    if len(paths) > 0:
        return paths[-1].resolve()
    else:
        logger.error("unable to get latest model: model folder is empty (are you sure a model has been trained)")
        sys.exit(1)


def format_output(dataframe, y):
    output_df = pd.DataFrame(data=dataframe)
    output_df['is_fake_probability'] = y
    output_df.drop("Event", axis=1, inplace=True)
    output_df.drop("Category", axis=1, inplace=True)
    return output_df


parser = argparse.ArgumentParser(description="This command line allows you to make\
a prediction given a .cvs file and an a trained model.")

parser.add_argument('--model', help='absolute path to mlflow model (MLmodel file) to use for prediction. \
    To specify when not using the latest model', dest="model")
parser.add_argument('--data', help='path to .csv input file to make prediction for',
                    dest="data", required=True)
parser.add_argument('--output', help='path to .csv where the result will be store',
                    dest="output", default="output.csv")
parser.add_argument('--latest', dest='latest', action='store_true', help='use latest model stored')
parser.add_argument('--no-latest', dest='latest', action='store_false', help='do not use latest model stored')
parser.set_defaults(latest=True)
args = parser.parse_args()


logging.info("Making prediction for %s with %s", args.data, args.model)
try:
    logging.info("Reading file at: %s", args.data)
    input_df = pd.read_csv(args.data)
except FileNotFoundError:
    logging.error("Please specify a valid file path for your input data")
    sys.exit(1)

original_df = input_df.copy()

try:
    input_df.drop("UserId", axis=1, inplace=True)
    ohe = joblib.load(config.ohe_features_path)
    logging.info("Featurization of input data")
    X = ohe.transform(input_df).toarray()
except FileNotFoundError:
    logging.error("Please specify a valid file path for the encoder")
    sys.exit(1)

logging.info("Making prediction")

try:
    if args.latest:
        latest_model = get_latest_model_dir(config.model_dir_path)
        model_uri = f"{latest_model}/artifacts/model"
        y = make_prediction(X, model_uri)
    else:
        print(args.model)
        if (args.model is None or args.model == ""):
            logging.error("Please provide a model directory")
            sys.exit(1)
        else:
            y = make_prediction(X, args.model)
   
except (IOError, ValueError) as e:
    logging.error("Invalid model path: %s", e)
    sys.exit(1)


output_df = format_output(original_df, y)
logging.info("output file at %s", args.output)
output_df.to_csv(args.output, index=False)
