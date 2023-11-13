"""
This file contains functions for input/output of csv files using pandas dataframes as substrate.
"""

import pathlib
import re
import datetime

import pandas as pd
import numpy as np

from .base import parse_dir_rcs_sess


def load_csv_df(
    p_data: str,
    f_data: str,
) -> pd.DataFrame:
    """Load a csv file into a pandas dataframe

    Args:
        p_data (pathlib.Path): absolute path to directory containing the datafile
        f_data (str): filename of data with extension

    Returns:
        pd.DataFrame: output pandas dataframe
    """
    # load the csv data and specify the header as first row
    df = pd.read_csv(str(pathlib.Path(p_data) / f_data), header=0)

    return df


def clean_dt_df(
    df: pd.DataFrame,
    dt_fmt: str = "%Y-%m-%d %H:%M:%S",
    tz_str: str = "America/Los_Angeles",
    bool_to_pandas: bool = True,
) -> pd.DataFrame:
    """Parse a pd.DataFrame and convert the datatime column in dataframe (might be string) to pd.Timestamp with
    timezone information

    Args:
        df (pd.DataFrame): pandas.Series format containing datetime string
        dt_fmt (str | None, optional): datetime format. Defaults to "%Y-%m-%d %H:%M:%S".
        tz_str (str, optional): tzinfo object for timezone. Defaults to "America/Los_Angeles".
        bool_to_pandas (bool, optional): Whether to convert into pandas for output. Defaults to True.

    Returns:
        pd.DataFrame: output pd.DataFrame with timezone-aware datetime data
    """
    # make copy of input dataframe
    df_out = df.copy()

    # obtain index of the time column
    idx_time_col = get_time_column(df)

    # convert the time to pd.Timestamp
    # first try to parse without timezone
    try:
        df_out["time"] = parse_dt_w_tz(
            df_out.iloc[:, idx_time_col],
            dt_fmt=dt_fmt,
            tz_str=tz_str,
            bool_to_pandas=bool_to_pandas,
        )

    # otherwise parse with no preset format
    except ValueError:
        df_out["time"] = parse_dt_w_tz(
            df_out.iloc[:, idx_time_col],
            dt_fmt=None,
            tz_str=tz_str,
            bool_to_pandas=bool_to_pandas,
        )

    return df_out


def parse_dt_w_tz(
    dt_str: pd.Series,
    dt_fmt: str | None,
    tz_str: str,
    bool_to_pandas: bool,
) -> pd.Series:
    """Parse a datetime string with a specified format and timezone

    Args:
        dt_str (pd.Series): pandas.Series format containing datetime string
        dt_fmt (str | None, optional): datetime format.
        tz_str (str, optional): tzinfo object for timezone.
        bool_to_pandas (bool, optional): Whether to convert into pandas for output.

    Returns:
        pd.Series: output pd.Series with timezone-aware datetime data
    """

    # first convert to pandas datetime
    dt_pd = pd.to_datetime(dt_str, format=dt_fmt)

    # if native and not time aware, localize to time zone
    if (
        dt_pd.iloc[0].tzinfo is None
        or dt_pd.iloc[0].tzinfo.utcoffset(dt_pd.iloc[0]) is None
    ):
        dt_pd = dt_pd.dt.tz_localize(tz_str)
    else:
        dt_pd = dt_pd.dt.tz_convert(tz_str)

    # convert to python datetime if desired
    dt_out = dt_pd if bool_to_pandas else dt_pd.to_pydatetime()

    return dt_out


def get_time_column(df: pd.DataFrame) -> int | ValueError:
    """Obtain column index of the time column in the dataframe

    Args:
        df (pd.DataFrame): input dataframe

    Raises:
        ValueError: if "time" or "localTime" not found in column headers

    Returns:
        int | ValueError: index of the time column
    """
    # obtain the column headers
    df_headers = df.columns

    # if renamed data (e.g. processed by Lauren's code)
    if "time" in df_headers:
        # get the index of the 'time' column
        idx_time_col = df_headers.get_loc("time")

    # otherwise if using original name
    elif "localTime" in df_headers:
        idx_time_col = df_headers.get_loc("localTime")

    # otherwise throw an exception
    else:
        raise ValueError("Time column not found in data")

    return idx_time_col


def convert_str_sess_to_pd(
    vec_p_file_in: list[str],
) -> pd.DataFrame:
    """Save the list of session paths to pd.DataFrame.

    Args:
        vec_p_file_in (list[str]): List of session paths.

    Raises:
        ValueError: More than one timestamp in session name

    Returns:
        pd.DataFrame: Pandas dataframe with the session paths.
    """

    df_out = pd.DataFrame()

    for p_file in vec_p_file_in:
        # parse the filenames to get the necessary information
        p_file = pathlib.Path(p_file)
        str_sess = p_file.name

        # now obtain the unix timestamp only
        vec_str_timestamp = re.findall(r"\d+", str_sess)
        if len(vec_str_timestamp) > 1:
            raise ValueError("More than one timestamp found in the session name.")
        float_timestamp = int(vec_str_timestamp[0]) / 1e3

        # now also convert the timestamp to a readable format
        str_timestamp = datetime.datetime.fromtimestamp(float_timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # now append to the dataframe
        df_out = df_out.append(
            {
                "Session": str_sess,
                "Timestamp": float_timestamp,
                "DateTime": str_timestamp,
            },
            ignore_index=True,
        )  # type: ignore

    # finally sort the rows based on the timestamp
    df_out.sort_values(by="Timestamp", inplace=True, ignore_index=True)

    return df_out


def parse_dir_df(
    p_dir_in: str,
) -> pd.DataFrame:
    """Parse a directory and return a pd.DataFrame with the session paths.

    Args:
        p_dir_in str: Path to the directory to parse.

    Returns:
        pd.DataFrame: Pandas dataframe with the session paths
    """

    vec_p_file_in = parse_dir_rcs_sess(p_dir_in)

    return convert_str_sess_to_pd(vec_p_file_in)