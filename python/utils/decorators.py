import polars as pl
import pandas as pd


def polarify_in(func):
    def wrapper(*args, **kwargs):
        df = args[0]
        if isinstance(df, pd.DataFrame):
            args[0] = pl.from_pandas(df)
        return func(*args, **kwargs)
    return wrapper


def pandify_in(func):
    def wrapper(*args, **kwargs):
        df = args[0]
        if isinstance(df, pl.DataFrame):
            args[0] = df.to_pandas()
        return func(*args, **kwargs)
    return wrapper


def polarify_out(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        else:
            return df
    return wrapper


def pandify_out(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        else:
            return df
    return wrapper