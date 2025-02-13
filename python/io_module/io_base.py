"""
This file contains basic functions useful for input/output of data.
"""

import pathlib
import glob

import numpy as np
import numpy.typing as npt
import polars as pl
import duckdb
import mat73
from .polars_io import load_database_pl, load_parquet_pl

import importlib


def load_data(config):
    """Load data into a polars dataframe

    Args:
        config (dict): dictionary containing the configuration parameters

    Returns:
        pl.DataFrame: output dataframe
    """
    # load the data
    if config["source"] == "database":
        df = load_database_pl(
            str_module=config["database_module"],
            path_database=config["database_path"],
            query=config["query"],
        )
    elif config["source"] == "parquet":
        df = load_parquet_pl(
            path_data=config["data_path"],
            kwargs=config.get("kwargs", {}),
        )
    else:
        raise ValueError("Data type not recognized or not implemented.")

    return df


def parse_dir_rcs_sess(
    p_dir_in: str,
) -> list[str]:
    """Parse a directory and return a list of all RCS session names in the directory.

    Args:
        p_dir_in (str): Path to the directory to parse.

    Raises:
        ValueError: Path does not exist.

    Returns:
        list[str]: ist of RCS session names
    """

    if not pathlib.Path(p_dir_in).exists():
        raise ValueError("Input directory does not exist.")

    return glob.glob(str(pathlib.Path(p_dir_in) / "[sS]ession**"))


def load_amp_gain_mat(
    p_amp_gain: str,
    f_amp_gain: str,
) -> npt.NDArray:
    """Load a mat file containing the amplifier gains for all channels

    Args:
        p_amp_gains (str): absolute path to directory containing the datafile
        f_amp_gains (str): filename of data with extension

    Returns:
        npt.NDArray: output numpy array containing amplifier gains
    """

    # load the mat file
    amp_gain = mat73.loadmat(str(pathlib.Path(p_amp_gain) / f_amp_gain))

    # extract the amplifier gains
    dict_amp_gain = amp_gain["ampGains"]
    amp_gain = np.stack([int(dict_amp_gain[f"Amp{i}"]) for i in range(1, 5)])

    return amp_gain
