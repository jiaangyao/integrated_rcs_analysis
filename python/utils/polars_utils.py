import polars as pl
import numpy as np


def extract_polars_column_as_ndarray(
    df: pl.DataFrame,
    column_name: str,
) -> np.ndarray:
    """Extract a column from a polars dataframe, and convert it to a numpy array.
    This function checks if the polars column is type List first. If so, converts
    to type Array so that selected column is returned as 2D array.
        This is necessary because pl.List types are converted to arrays of array objects when .to_numpy() is called.
    Note, the returned array is transposed so that the first dimension is the number of rows, and the second dimension
    is the width the list (e.g array) in each row in the original dataframe

    Args:
        df (pl.DataFrame): input dataframe
        column_name (str): name of the column to extract

    Returns:
        np.ndarray: output numpy array
    """

    if isinstance(df.get_column(column_name).dtype, pl.List):
        list_width = df.select(pl.col(column_name).list.len()).unique()
        if list_width.height > 1:
            raise ValueError(
                "Column contains lists of different lengths. Cannot convert to 2D array"
            )
        else:
            width = list_width.item()

        return (
            df.select(pl.col(column_name).cast(pl.Array(inner=pl.Float64, width=width)))
            .to_numpy()
            .T
        )
    else:
        return df.get_column(column_name).to_numpy()
