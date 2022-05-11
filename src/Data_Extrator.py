from pandas import DataFrame

import pandas as pd


def read_csv(path: str, sep: str = ";") -> DataFrame:
    return pd.read_csv(path, sep=sep)


def create_df(data: list[dict] = None, columns: list[str] = None) -> DataFrame:
    return DataFrame(columns=columns, data=data)


def append_data(df: DataFrame, data: dict) -> DataFrame:
    return df.append(data, ignore_index=True)


def concat_data(df: DataFrame, data: DataFrame) -> DataFrame:
    return pd.concat([df, data], ignore_index=True)


def normalize_df(df: DataFrame) -> DataFrame:
    return (df - df.min()) / (df.max() - df.min())
