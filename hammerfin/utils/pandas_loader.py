import os

import pandas as pd


def load_data(file_path, **kwargs):
    """
    Load data into pandas DataFrame from a file path

    Supported extensions are .csv, .xls, .xlsx, .json, .pkl, .feather, .parquet
    """

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".csv":
        return pd.read_csv(file_path, **kwargs)
    if file_extension in [".xls", ".xlsx"]:
        return pd.read_excel(file_path, **kwargs)
    if file_extension == ".json":
        return pd.read_json(file_path, **kwargs)
    if file_extension == ".pkl":
        return pd.read_pickle(file_path, **kwargs)
    if file_extension == ".feather":
        return pd.read_feather(file_path, **kwargs)
    if file_extension == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    raise ValueError(
        f"""Unsupported file extension: {file_extension}.
Supported extensions are .csv, .xls, .xlsx, .json, .pkl, .feather, .parquet"""
    )
