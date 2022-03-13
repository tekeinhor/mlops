from src.processor import DataProcessor
import pandas as pd

processor = DataProcessor("")


def test_preprocess():
    data = {'UserId': ['F7A7BF3761', 'BA8F7A71E6'],
            'Event': ['click_carrousel', 'send_sms'],
            'Category': ['Phone', 'Motor'],
            'Fake': [0, 0]}
    df = pd.DataFrame(data)
    x, y = processor.preprocess(df)

    assert len(x) == len(df)
    assert len(y) == len(df)
    assert len(x.columns) == 2
    return


def test_clean():
    data = {'Unnamed: 0': [0, 1],
            'UserId': ['F7A7BF3761', 'BA8F7A71E6'],
            'Event': ['click_carrousel', 'send_sms'],
            'Category': ['Phone', 'Motor'],
            'Fake': [0, 0]}
    df = pd.DataFrame(data)
    df_size = len(df)
    df_column = len(df.columns)

    cleaned_df = processor.clean(df)
    assert len(cleaned_df) == df_size
    assert len(cleaned_df.columns) == df_column - 1
    return


def test_load_data():
    return
