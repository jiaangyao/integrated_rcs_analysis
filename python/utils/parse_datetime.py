import pandas as pd


def parse_dt_w_tz(
    dt_str,
    dt_fmt="%Y-%m-%d %H:%M:%S",
    tz_str="America/Los_Angeles",
    to_pandas=True,
):
    """
    Parse a datetime string with a specified format and timezone

    :param pd.Series dt_str: pandas.Series format containing datetime string
    :param dt_fmt: datetime format
    :param tz_str: tzinfo object for timezone

    :return: converted datetime object
    """

    dt_pd = pd.to_datetime(dt_str, format=dt_fmt)
    dt_pd = dt_pd.dt.tz_localize(tz_str)

    dt_out = dt_pd if to_pandas else dt_pd.to_pydatetime()

    return dt_out
