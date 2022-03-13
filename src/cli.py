import click
import sys
import pandas as pd
from src.engine import Engine
import logging
import coloredlogs
from src.config import settings
from src.processor import DataProcessor
coloredlogs.install(level='INFO', fmt='%(asctime)s,%(msecs)03d:(%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)


def format_output(dataframe, y):
    output_df = pd.DataFrame(data=dataframe)
    output_df['is_fake_probability'] = y
    output_df.drop("Event", axis=1, inplace=True)
    output_df.drop("Category", axis=1, inplace=True)
    return output_df


@click.group()
def cli():
    """
    CLI for training and predicting.
    """
    pass


@cli.command()
@click.option('--name', prompt_required=False,
              help='registered model name to use for prediction.')
@click.option('--version', prompt_required=False,
              help='registered model version to use for prediction.')
@click.option('--data', help='path to .csv input file to make prediction for.')
@click.option('--output', help='path to .csv where the result will be store.', default=settings.DEFAULT_OUTPUT)
def predict(data: str, name: str, version: str, output: str):
    """Predict command for cli"""
    # open data
    try:
        logger.info("Reading file at: %s", data)
        input_df = pd.read_csv(data)
        original_df = input_df.copy()
    except FileNotFoundError:
        logging.error("Please specify a valid file path for your input data")
        sys.exit(1)

    # create engine
    engine = Engine(settings.ENCODER_PATH, settings.MLFLOW.TRACKING_URI)

    # perform prediction
    try:
        model_uri = f"models:/{name}/{version}"
        X = engine.featurize(input_df)
        y = engine.predict(X, model_uri)
    except Exception as e:
        logger.error("Unable to perform prediction model: %s", e)
        sys.exit(1)

    output_df = format_output(original_df, y)

    # output to csv
    logger.info("storing result at : %s", output)
    output_df.to_csv(output, index=False)


@cli.command()
@click.option('--trainset', help='path to .csv train file.')
@click.option('--testset', help='path to .csv test file.')
@click.option('--sampling', is_flag=True, help='option to perform sampling before training.')
def train(trainset: str, testset: str, sampling: bool):
    """Train command for cli"""
    logger.info("TRAIN: %s, TEST: %s, SAMPLING: %s", trainset, testset, sampling)

    # create data processor
    processor = DataProcessor(settings.ENCODER_PATH)

    # perform data cleaning, processing, sampling
    try:
        x_train, y_train, y_train,  x_test, y_test = processor.create(trainset, testset, sampling)
    except Exception as e:
        logger.exception("Unable to create data: %s", e)
        sys.exit(0)

    engine = Engine(settings.ENCODER_PATH, settings.MLFLOW.TRACKING_URI)
    run_id, artifact_path, registered_model_name = engine.train(x_train, y_train,  x_test, y_test)

    if click.confirm('The training is done, do you want to register this model?'):
        engine.register_model(run_id, artifact_path, registered_model_name)
    else:
        logger.info("Model not registered.")


@cli.command()
def list_models():
    """List registered model available for prediction."""
    engine = Engine(settings.ENCODER_PATH, settings.MLFLOW.TRACKING_URI)
    if click.confirm('Do you know the model name ?'):
        model_name = click.prompt('Please enter the model name', type=str)
        engine.get_model_versions(model_name)
    else:
        engine.get_models()


if __name__ == "__main__":
    cli()
