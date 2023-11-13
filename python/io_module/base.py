import pathlib
import glob
import polars as pl
import duckdb

import importlib


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
  