"""
This file contains functions for input/output of database/parquet data files using polars.
"""

import pathlib
import importlib

import polars as pl


def load_database_pl(
    str_module: str,
    path_database: str,
    query: str,
) -> pl.DataFrame:
    """Load data from a database into a polars dataframe.

    Args:
        str_module (str): module of the database, e.g. "duckdb"
        database_path (str): absolute path to the database
        query (str): SQL query to execute

    Returns:
        pl.DataFrame: output dataframe
    """

    module = importlib.import_module(str_module)

    con = module.connect(path_database, read_only=True)
    df = con.sql(query).pl()
    con.close()
    return df


def load_parquet_pl(
    path_data: str,
) -> pl.DataFrame:
    """Load a parquet file into a polars dataframe

    Args:
        data_path (str): absolute path to data file

    Returns:
        pl.DataFrame: output dataframe
    """
    # load the csv data and specify the header as first row
    df = pl.read_parquet(path_data)

    return df