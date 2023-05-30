import pathlib
import glob
import re
import datetime

import pandas as pd


def parse_directory_rcs_sess(
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


def convert_str_sess_to_pd(
    vec_p_file_in: list[str],
) -> pd.DataFrame:
    """Save the list of session paths to a pandas dataframe.

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


def parse_directory_rcs_sess_to_pd(
    p_dir_in: str,
) -> pd.DataFrame:
    """Parse a directory and return a pandas dataframe with the session paths.

    Args:
        p_dir_in str: Path to the directory to parse.

    Returns:
        pd.DataFrame: Pandas dataframe with the session paths
    """

    vec_p_file_in = parse_directory_rcs_sess(p_dir_in)

    return convert_str_sess_to_pd(vec_p_file_in)


if __name__ == "__main__":
    p_rcs_in = "/home/jyao/local/data/starrlab/raw_data/RCS17/RCS17R/"
    p_csv_out = "/home/jyao/local/data/starrlab/raw_data/RCS17/RCS17R/"
    f_csv_out = "RCS17_R_sess_list.csv"

    # parse the directory
    vec_p_sess = parse_directory_rcs_sess(p_rcs_in)
    df_sess = convert_str_sess_to_pd(vec_p_sess)

    # now save the output
    df_sess.to_csv(str(pathlib.Path(p_csv_out) / f_csv_out), index=False)
