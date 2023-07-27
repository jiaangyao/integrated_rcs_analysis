import pandas as pd


def parse_dt(dt_str, **kwargs):
    # first try to parse without timezone
    try:
        dt_out = parse_dt_w_tz(dt_str, **kwargs)

    # otherwise parse with datetime information in mind
    except ValueError:
        dt_out = parse_dt_w_tz(
            dt_str,
            dt_fmt=None,
            tz_str=kwargs["tz_str"],
            bool_to_pandas=kwargs["bool_to_pandas"],
        )

    return dt_out


def parse_dt_w_tz(
    dt_str: pd.Series,
    dt_fmt: str | None = "%Y-%m-%d %H:%M:%S",
    tz_str: str = "America/Los_Angeles",
    bool_to_pandas: bool = True,
):
    """ Parse a datetime string with a specified format and timezone

    Args:
        dt_str (pd.Series): pandas.Series format containing datetime string
        dt_fmt (str | None, optional): datetime format. Defaults to "%Y-%m-%d %H:%M:%S".
        tz_str (str, optional): tzinfo object for timezone. Defaults to "America/Los_Angeles".
        bool_to_pandas (bool, optional): Whether to convert into pandas for output. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    # first convert to pandas datetime
    dt_pd = pd.to_datetime(dt_str, format=dt_fmt)
    
    # if native and not time aware, localize to time zone
    if dt_pd.iloc[0].tzinfo is None or dt_pd.iloc[0].tzinfo.utcoffset(dt_pd.iloc[0]) is None:
        dt_pd = dt_pd.dt.tz_localize(tz_str)
    else:
        dt_pd = dt_pd.dt.tz_convert(tz_str)

    # convert to python datetime if desired
    dt_out = dt_pd if bool_to_pandas else dt_pd.to_pydatetime()

    return dt_out


def get_time_column(data_hemi):
    # obtain the column headers
    data_hemi_header = data_hemi.columns

    # if renamed data (e.g. processed by Lauren's code)
    if "time" in data_hemi_header:
        # get the index of the 'time' column
        idx_time_col = data_hemi_header.get_loc("time")

    # otherwise if using original name
    elif "localTime" in data_hemi_header:
        idx_time_col = data_hemi_header.get_loc("localTime")

    # otherwise throw an exception
    else:
        raise ValueError("Time column not found in data")

    return idx_time_col
