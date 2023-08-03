"""
This file contains general wrapper functions for handling general input/output of data.
"""
import pathlib

import pandas as pd

from .pandas_io import load_csv_df, clean_dt_df, extract_td_data_df

# TODO: enforce typing for output variables

def load_df(
    p_data: pathlib.Path,
    f_data: str,
):
    """Load dataframe from file using pandas or polars

    Args:
        p_data (pathlib.Path): absolute path to directory containing the datafile
        f_data (str): filename of data with extension

    Raises:
        NotImplementedError: Polars is not implemented yet

    Returns:
        _type_: output dataframe
    """    
    # right now does simple testing of whether file is csv or no
    # if csv then use pandas to load
    if f_data.endswith(".csv"):
        df = load_csv_df(p_data, f_data)

    # otherwise throw exception for now
    else:
        # TODO: need to implement case for polars
        raise NotImplementedError(
            "Loading of dataframe using parque and polars is not implemented yet"
        )

    return df


def correct_df_time(df):
    """Convert the time column in dataframe (might be string) to datetime data with timezone information

    Args:
        df (_type_): input dataframe

    Raises:
        NotImplementedError: Polars is not implemented yet

    Returns:
        _type_: output dataframe with a column named "time" that has datetime data with timezone information
    """    
    # if dataframe is pandas dataframe, then correct the time column
    if isinstance(df, pd.DataFrame):
        df = clean_dt_df(df)

    # otherwise throw exception for now
    else:
        raise NotImplementedError("Only pandas dataframe is supported for now")

    return df


def extract_td_data(df, label_type: str):
    # if dataframe is pandas dataframe, then extract the time domain data only
    if isinstance(df, pd.DataFrame):
        data_td, label_td, fs, str_ch, ch2use = extract_td_data_df(df, label_type)

    # otherwise throw exception for now
    else:
        raise NotImplementedError("Only pandas dataframe is supported for now")

    return data_td, label_td, fs, str_ch, ch2use


def load_data(
    p_data: pathlib.Path,
    f_data: str,
    label_type: str,
):
    """Load data from file using pandas or polars

    Args:
        p_data (pathlib.Path): absolute path to directory containing the datafile
        f_data (str): filename of data with extension
        label_type (str): type of label to use

    Raises:
        NotImplementedError: Polars is not implemented yet

    Returns:
        _type_: output time domain data and label
    """
    # load the data from the designated path
    df = load_df(p_data, f_data)

    # quickly convert the time to datetime format
    df = correct_df_time(df)

    # next extract the time domain data only
    data_td, label_td, fs, str_ch, ch2use = extract_td_data(df, label_type)

    return data_td, label_td, fs, str_ch, ch2use