import pathlib
import glob
import polars as pl
import duckdb

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


def load_data(data_params):
    """
    Load data from a file or database.
    """
    if data_params['source'] == 'database':
        con = duckdb.connect(data_params['database_path'], read_only=True)
        df = con.sql(data_params['query']).pl()
        con.close()
        return df
    else:
        return pl.read_parquet(data_params['data_path'])