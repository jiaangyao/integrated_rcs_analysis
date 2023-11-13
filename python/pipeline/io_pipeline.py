"""
This file contains general wrapper functions for handling general input/output of data.
"""
import pathlib

import pandas as pd
import polars as pl

from io_module.pandas_io import load_csv_df, clean_dt_df
from io_module.polars_io import load_database_pl, load_parquet_pl


def load_df(
    data_params: dict,
) -> pd.DataFrame | pl.DataFrame:
    """Load a datafile into a pandas or polars dataframe

    Args:
        data_source_params (dict): list of parameters for loading data, loaded from yaml file

    Returns:
        pd.DataFrame | pl.DataFrame: output dataframe
    """
    # right now does simple testing of whether file is csv or no
    # if csv then use pandas to load
    if data_params["file_data"].endswith(".csv"):
        df = load_csv_df(data_params["path_data"], data_params["file_data"])

    # otherwise use polars to load
    else:
        # importing a database
        if data_params["source"] == "database":
            df = load_database_pl(
                data_params["database_module"],
                data_params["database_path"],
                data_params["query"],
            )
        # importing a parquet file
        else:
            df = load_parquet_pl(data_params["data_path"])

    return df


def correct_df_time(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """Convert the time column in dataframe (might be string) to datetime data with timezone information

    Args:
        df (pd.DataFrame | pl.DataFrame): input dataframe

    Returns:
        pd.DataFrame | pl.DataFrame: output dataframe with a column named "time" that has datetime data with timezone information
    """

    # if dataframe is pandas dataframe, then correct the time column
    # and corrected
    if isinstance(df, pd.DataFrame):
        df = clean_dt_df(df)
        df_pl = pl.from_pandas(df)

    # otherwise don't do any datetime corection
    # TODO: @claysmyth implement this for polars
    else:
        pass

    return df_pl


def load_data(
    data_source_params: dict,
) -> pl.DataFrame:
    """Load data from file using pandas or polars

    Args:
        data_source_params (dict): list of parameters for loading data, loaded from yaml file

    Returns:
        pl.DataFrame: output polars dataframe with corrected time column
    """

    # load the data from the designated path
    df = load_df(data_source_params)

    # quickly convert the time to datetime format
    df = correct_df_time(df)

    return df
