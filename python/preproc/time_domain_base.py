"""
This function contains base functionalities for preprocessing data in time domain. (Previously preproc/time_domain_processing)
"""
from typing import List

import polars as pl
import polars.selectors as cs
import pandas as pd
from typing import List, Union, Dict
from ._pl_namespace.filt import butterworth_bandpass_np


def identity(X):
    """
    Identity function. Returns input.
    Helpful if no feature engineering is desired.
    """
    return X


def standardize_df_columns(df, cols_to_standardize):
    """
    Center to mean=0 and scale variance=1 for the specified columns in a DataFrame.
    params:
    - df (polars.DataFrame): The DataFrame to be standardized.
    - cols_to_standardize (List[str]): A list of column names to standardize.

    returns:
    - polars.DataFrame: A new DataFrame with the specified columns standardized (in place).
    """
    return df.with_columns(
        [
            (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
            for col in cols_to_standardize
        ]
    )


def normalize_df_columns(df, cols_to_standardize, range=[0, 1]):
    """
    Normalize the specified columns in a DataFrame to have the range provided.
    params:
    - df (polars.DataFrame): The DataFrame to be standardized.
    - cols_to_standardize (List[str]): A list of column names to normalize.
    - range (List[float]): A list of length 2 specifying the range to normalize to. Default is [0,1].
    See: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    returns:
    - polars.DataFrame: A new DataFrame with the specified columns normalized (in place).
    """
    return df.with_columns(
        [
            (pl.col(col) - pl.col(col).min())
            / (pl.col(col).max() - pl.col(col).min())
            * (range[1] - range[0])
            + range[0]
            for col in cols_to_standardize
        ]
    )


def epoch_df_by_timesegment(
    df: pl.DataFrame,
    interval: str = "1s",
    period: str = "2s",
    sample_rate: int = 500,
    align_with_PB_outputs: bool = False,
    td_columns: List[str] = ["TD_BG", "TD_key2", "TD_key3"],
    sort_by_col="localTime",
    group_by_cols: List[str] = ["SessionIdentity"],
    scalar_cols: List[str] = [],
    vector_cols: List[str] = [],
) -> pl.DataFrame:
    """
    Epoch a DataFrame based on a time interval and period.

    Parameters:
    - df (polars.DataFrame): The DataFrame to be epoched.
    - interval (str): The time interval between the start of each time segment in seconds. Default is '1s'.
    - period (str): The length of each time segment in seconds. Default is '2s'.
    - sample_rate (int): The sampling rate of the data. Used to calculate the number of samples in each time segment. Default is 500.
    - align_with_PB_outputs (bool): If True, the time segments will be aligned with the Power Band outputs. Default is False.
    - td_columns (List[str]): A list of raw time domain columns to include in the resulting DataFrame. Default is ['TD_BG', 'TD_key2', 'TD_key3'].
    - sort_by_cols (str): Column by which windowing is performed. Default is 'localTime'. Needs to be a datetime column.
    - group_by_cols (List[str]): A list of columns to group by. Default is ['SessionIdentity'].
    - scalar_cols (List[str]): A list of columns to include in the resulting DataFrame, where a single scalar value, the last value in the aggregation, is extracted after epoching. Default is [].
    - vector_cols (List[str]): A list of columns to include in the resulting DataFrame, where the aggregation creates a vector for the column values within each epoched window. Default is [].
    # TODO: Consider including kwarg that is a list of functions to apply to column subset, e.g. [pl.col(col).mean().alias(f'{col}_mean') for col in td_columns]

    Returns:
    - polars.DataFrame: A new DataFrame with the specified columns and epoched time segments.
    """

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # TODO: Consider 'streaming' option to save on RAM

    if align_with_PB_outputs:
        df_pb_count = (
            df.join(
                df.filter(pl.col("Power_Band8").is_not_null())
                .select("DerivedTime")
                .with_row_count(),
                on="DerivedTime",
                how="left",
            )
            .with_columns(pl.col("row_nr").fill_null(strategy="backward"))
            .rename({"row_nr": "PB_count"})
        )

        num_windows_in_each_period = int(period[:-1]) // int(interval[:-1])
        df_pb_count = df_pb_count.with_columns(
            [
                pl.when((pl.col("PB_count") % num_windows_in_each_period) != i)
                .then(pl.lit(None))
                .otherwise(pl.col("PB_count"))
                .fill_null(strategy="backward")
                .alias(f"PB_count_{i}")
                for i in range(num_windows_in_each_period)
            ]
        )

        # NOTE: Windows are likely not in chronological order
        df_epoched = (
            pl.concat(
                [
                    df_pb_count.group_by(group_by_cols + [f"PB_count_{i}"])
                    .agg(
                        [pl.col(td_col) for td_col in td_columns]
                        + [pl.col(td_columns[0]).count().alias("TD_count")]
                        + [pl.col(col) for col in vector_cols]
                        + [pl.col(col).drop_nulls().first() for col in scalar_cols]
                    )
                    .rename({f"PB_count_{i}": "PB_ind"})
                    for i in range(num_windows_in_each_period)
                ],
                how="vertical",
            )
            .select(pl.all().shrink_dtype())
            .rechunk()
        )

    else:
        epoch_length = int(period[:-1]) * sample_rate
        df_epoched = (
            df.sort(sort_by_col)
            .group_by_dynamic(
                sort_by_col, every=interval, period=period, by=group_by_cols
            )
            .agg(
                [pl.col(td_col) for td_col in td_columns]
                + [pl.col(td_col).count().suffix("_TD_count") for td_col in td_columns]
                + [pl.col(col).suffix("_vec") for col in vector_cols]
                + [pl.col(col).drop_nulls().last() for col in scalar_cols]
            )
            .select(pl.all().shrink_dtype())
        )

        df_epoched = (
            df_epoched.with_columns(
                [
                    pl.col(td_col)
                    .list.eval(pl.element().is_not_null())
                    .list.all()
                    .suffix("_contains_null")
                    for td_col in td_columns
                ]
                # Remove rows where the TD data is null, or where the TD data is not the correct length
            )
            .filter(
                (pl.all_horizontal(pl.col("^.*_TD_count$") == epoch_length))
                & (pl.all_horizontal("^.*_contains_null$"))
            )
            .with_columns(
                [
                    pl.col(col).cast(pl.Array(width=epoch_length, inner=pl.Float64))
                    for col in td_columns
                ]
            )
            .select(pl.all().exclude("^.*TD_count$"))
            .select(pl.all().exclude("^.*_contains_null$"))
        )

    return df_epoched


def bandpass_filter(
    df: pl.DataFrame, columns: List[str], filt_args: Dict, group_by=[]
) -> pl.DataFrame:
    """
    Apply a bandpass filter to the specified columns in a DataFrame.
    params:
    - df (polars.DataFrame): The DataFrame to be filtered.
    - columns (List[str]): A list of column names to apply the filter.
    - filt_args (Dict): Dictionary of list or tuple of arguments to pass to scipy.signal.butter for the bandpass filter.
        Each key-value pair is applied to each column in 'columns'. Each Dictionary key should be the desired suffix name for new column, and each value should be a list or tuple of arguments to pass to scipy.signal.butter.
        Tuple | List Parameters for scipy.signal.butter:
            Nint
            The order of the filter. For bandpass and bandstop filters, the resulting order of the final second-order sections (sos) matrix is 2*N, with N the number of biquad sections of the desired system.

            Wn
            The critical frequency or frequencies. For bandpass and bandstop filters, Wn is a length-2 sequence.

            For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).

            For digital filters, if fs is not specified, Wn units are normalized from 0 to 1, where 1 is the Nyquist frequency (Wn is thus in half cycles / sample and defined as 2*critical frequencies / fs). If fs is specified, Wn is in the same units as fs.

            For analog filters, Wn is an angular frequency (e.g. rad/s).

            fsfloat
            The sampling frequency of the digital system.
    - group_by (List[str]): A list of columns to group by. Default is []. (E.g. group_by=['SessionIdentity'] or group_by=['SessionDate'])

    returns:
    - polars.DataFrame: A new DataFrame with the specified columns filtered as new columns, with suffix. The Suffix is the key from the filt_args dictionary.
    """
    # First, chunk together sections of non-null and null values,
    # to avoid introducing artifacts at the boundaries of null values.
    # Use row count (row_nr) to keep track of row order, e.g. as index column
    pl_cols = cs.by_name(*columns)
    df = (
        df.with_row_count()
        .set_sorted("row_nr")
        .with_columns(
            # Check which values are null or nan
            # pl.concat_list(*columns).is_null().any() OR pl.any_horizontal(pl_cols.is_null())
            (
                pl.when(
                    pl.any_horizontal(pl_cols.is_null())
                    | pl.any_horizontal(pl_cols.is_nan())
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                # Check to see when values change from null or nan to not null or nan, and vice versa
                .diff().fill_null(0).abs()
                # Assign a row number to each row where there is a switch from null to non-null, or vice versa
                * pl.col("row_nr")
            )
            # Fill 0's with previous row number to get a unique identifier for each null/non-null segment
            .cumsum().alias("null_seg_nr")
            # Drop Null Rows, i.e. where columns are null (usually due to disconnects)
        )
        .filter(
            pl.all_horizontal(pl_cols.is_not_null())
            & pl.all_horizontal(pl_cols.is_not_null())
        )
    )

    # Call the filtering function on each desired column and time segment, avoiding null sections
    df_filt = (
        df.group_by(group_by + ["null_seg_nr"]).agg(
            [
                pl.col(col).filt.butterworth_bp(N, Wn, fs).suffix(f"_{key}")
                # Note: can also use map_elements on the numpy array directly
                # pl.col(col).map_elements(lambda x: butterworth_bandpass_np(x.to_numpy(), N, Wn, fs)).suffix(f'_{key}')
                for key, (N, Wn, fs) in filt_args.items()
                for col in columns
            ]
            + [pl.col("row_nr")]
        )
        # Remove rows where the filter returned null
        .filter(pl.all_horizontal(pl.all().is_not_null()))
        # Remove unnecessary columns
        .drop(group_by + ["null_seg_nr"])
        # Convert back to long format
        .explode(
            ["row_nr"] + [f"{col}_{key}" for key in filt_args.keys() for col in columns]
        )
        # Sort back into chronological order
        .sort("row_nr")
    )

    # Join as new columns into original DataFrame
    return df.join(df_filt, on="row_nr", how="left").drop(["row_nr", "null_seg_nr"])
