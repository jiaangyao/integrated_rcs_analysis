import polars as pl
import polars.selectors as cs
import re
from typing import Union, List
from utils.decorators import polarify_in

data_type_map = {
    "pl.Int8": pl.Int8,
    "pl.Int16": pl.Int16,
    "pl.Int32": pl.Int32,
    "pl.Int64": pl.Int64,
    "pl.UInt8": pl.UInt8,
    "pl.UInt16": pl.UInt16,
    "pl.UInt32": pl.UInt32,
    "pl.UInt64": pl.UInt64,
    "pl.Float32": pl.Float32,
    "pl.Float64": pl.Float64,
    "pl.Boolean": pl.Boolean,
    "pl.Utf8": pl.Utf8,
    "pl.Date": pl.Date,
    "pl.Datetime": pl.Datetime,
    "pl.Time": pl.Time,
    "pl.Duration": pl.Duration,
    "pl.Object": pl.Object,
    "pl.Categorical": pl.Categorical,
    "pl.List": pl.List,
    "pl.Struct": pl.Struct,
    "pl.INTEGER_DTYPES": pl.INTEGER_DTYPES,
    "pl.FLOAT_DTYPES": pl.FLOAT_DTYPES,
    "pl.DATETIME_DTYPES": pl.DATETIME_DTYPES,
    "pl.DURATION_DTYPES": pl.DURATION_DTYPES,
    "pl.NUMERIC_DTYPES": pl.NUMERIC_DTYPES,
    # 'pl.Dict': pl.Dict,  # Uncomment if this type is available in your Polars version
}

selectors_map = {
    "cs.categorical": cs.categorical(),
    "cs.date": cs.date(),
    "cs.datetime": cs.datetime(),
    "cs.duration": cs.duration(),
    "cs.float": cs.float(),
    "cs.integer": cs.integer(),
    # 'cs.matches': cs.matches(), needs argument
    "cs.numeric": cs.numeric(),
    "cs.string": cs.string(),
    "cs.temporal": cs.temporal(),
    "cs.time": cs.time(),
}


def rename_columns(df, rename_dict={}):
    """Rename columns in a polars DataFrame

    Args:
        df (pl.DataFrame): input polars DataFrame
        rename_dict (dict): dictionary with old column name as key and new column name as value

    Returns:
        pl.DataFrame: output polars DataFrame with renamed columns
    """
    return df.rename(rename_dict)


# @polarify_in
def select_columns(
    df: pl.DataFrame,
    columns: List[str] = None,
    regex: str = None,
    dtype: Union[List[str], str] = None,
    selectors: List[str] = None,
    exclude: List[str] = None,
) -> pl.DataFrame:
    """
    Select columns from a polars DataFrame using a list of column names, a regex pattern, a list of Polars DataTypes, or a list of Polars selectors.

    args:
        df (pl.DataFrame): input polars DataFrame
        columns (List[str]): list of column names to select
        regex (str): regex pattern to select columns
        dtype (Union[List[str], str]): list of Polars DataTypes to select. Will be converted to Polars DataTypes using the data_type_map
        selectors (List[str]): list of Polars selectors to select. Will be converted to Polars selectors using the selectors_map
        exclude (List[str]): list of column names to exclude
        TODO: Include things like column index, column names end/start/match a pattern, etc...

    returns:
        pl.DataFrame: output polars DataFrame with selected columns
    """

    # May need debugging

    cols_to_keep = []

    # Column selection using polars selectors
    if columns:
        # Selector is a list of column names
        cols_to_keep.append(cs.by_name(*columns))

    if regex:
        # Selector is a regex pattern
        cols_to_keep.append(cs.matches(regex))

    if dtype:
        if isinstance(dtype, str):
            dtype = [dtype]
        dtype = (
            [data_type_map[d] for d in dtype]
            if isinstance(dtype, list)
            else data_type_map[dtype]
        )
        cols_to_keep.append(cs.by_type(*dtype))

    if selectors:
        selectors = [selectors_map[s] for s in selectors]
        cols_to_keep.append(*selectors)

    if cols_to_keep:
        df = df.select(*cols_to_keep)

    # Columns to exclude
    if exclude is not None:
        df = df.select(pl.all().exclude(exclude))
    else:
        raise ValueError(
            "Selector must be a list of column names, a list of Polars DataTypes, or a regex pattern."
        )

    return df


# @polarify_in
def filter_rows(df: pl.DataFrame, condition: str = "") -> pl.DataFrame:
    """
    Filter rows in a polars DataFrame using a condition string.

    args:
        df (pl.DataFrame): input polars DataFrame
        condition (str or List[str]): condition string to filter rows. Can be arbitrarily complex, as long as it is well-constructed and a valid Python expression. Newlines may cause issues.
            E.g. "(pl.col('A') > 0" | "pl.col('A') < -10) & (pl.col('B') == 'taco') & (pl.col('C').is_in([1, 2, 3]))"

    returns:
        pl.DataFrame: output polars DataFrame with filtered rows
    """

    if condition:
        condition = eval(condition)

        return df.filter(condition)

    else:
        return df


def with_columns(df: pl.DataFrame, operations: List[str] = None) -> pl.DataFrame:
    """
    Apply operations to columns in a polars DataFrame.

    args:
        df (pl.DataFrame): input polars DataFrame
        operations (List[str]): list of operations to apply to columns. Each operation should be a string that can be evaluated as a valid Python expression. May be able to use () for multi-line or particulalry complex operations.
            E.g. ["C = pl.col('A') + pl.col('B')", "pl.col('A').is_null().alias('A_null'), "pl.when(pl.col('foo') > 2).then(1).otherwise(-1).alias('val')"]

    returns:
        pl.DataFrame: output polars DataFrame with modified columns
    """

    if operations:
        operations = [eval(op) for op in operations]

        return df.with_columns(*operations)

    else:
        return df
