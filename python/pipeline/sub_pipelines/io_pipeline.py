"""
This file contains general wrapper functions for handling general input/output of data.
"""
import pathlib

import pandas as pd
import polars as pl

from io_module.io_base import load_amp_gain_mat
from io_module.pandas_io import load_csv_df, clean_dt_df
from io_module.polars_io import load_database_pl, load_parquet_pl
from utils.decorators import polarify_out


def load_df_from_file(
    file_type: str,
    data_path: str,
    kwargs: dict = None,
) -> pd.DataFrame | pl.DataFrame:
    """Load a datafile into a pandas or polars dataframe

    Args:
        data_source_params (dict): list of parameters for loading data, loaded from yaml file

    Returns:
        pd.DataFrame | pl.DataFrame: output dataframe
    """
    # right now does simple testing of whether file is csv or no
    # if csv then use pandas to load
    if file_type == "csv":
        # df = load_csv_df(
        #     data_source_params["path_data"], data_source_params["file_data"]
        # )
        df = pl.read_csv(data_path, **kwargs)
    # otherwise use polars to load
    elif file_type == "parquet":
        df = pl.read_parquet(data_path, **kwargs)
    elif file_type == "mat":
        raise NotImplementedError
    else:
        print("File type not supported")
        raise NotImplementedError

    return df


@polarify_out
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
        # df_pl = pl.from_pandas(df)

    # otherwise don't do any datetime corection
    # TODO: @claysmyth implement this for polars
    else:
        pass

    return df


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
    if data_source_params["source"] == "database":
        df = load_database_pl(
            data_source_params["database_module"],
            data_source_params["database_path"],
            data_source_params["query"],
        )
    elif (
        data_source_params["source"] == "parquet"
        or data_source_params["source"] == "csv"
        or data_source_params["source"] == "mat"
    ):
        kwargs = data_source_params.get("kwargs", {})
        df = load_df_from_file(
            data_source_params["source"], data_source_params["data_path"], kwargs
        )

    # quickly convert the time to datetime format
    df = correct_df_time(df)

    return df


def load_amp_gain(
    data_source_params: dict,
    mode: str = "mat",
) -> dict:
    """Load the amplifier gain file

    Args:
        data_source_params (dict): list of parameters for loading data, loaded from yaml file
        mode (str, optional): type of file provided. Defaults to "mat".

    Returns:
        dict: amplifier gain dictionary
    """

    # load the amplifier gain file
    if mode == "mat":
        amp_gain = load_amp_gain_mat(
            data_source_params["path_data"], data_source_params["file_gain"]
        )
    elif mode == "null":
        amp_gain = None
    else:
        raise NotImplementedError

    return amp_gain
