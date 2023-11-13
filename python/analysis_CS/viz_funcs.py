import plotly.graph_objects as go
from plotly.colors import n_colors
from collections import OrderedDict
import polars as pl
from utils.decorators import polarify_in
import wandb

@polarify_in
def plot_feature_distributions(
    df, melt_cols, partition_col, value_name="Value", color_ordering=None, row_ordering=None
):
    """
    Plots the distributions of each desired column (melt_cols) in df by values in partition_col.
    parameters:
    df: polars dataframe.
    melt_cols: (list of strings) columns to melt. These columns will ultimate be rows in the resulting ridge plot.
    partition_col: (string) column to partition by. A separate distribution of each melt_col for each unique value in the partition_col will be plotted.
    value_name: (string) name of the resulting value column. This is the x-axis label in the ridge plot.
    color_ordering: (list of strings) order of the colors in the ridge plot. The elements in this list should correspond to unique values in the partition_col.
                    If None, the colors will be assigned in the order of the unique values in partition_col.
    row_ordering: (list of strings) order of the rows in the ridge plot, from bottom to top. The elements in this list should correspond to unique values in the melt_cols.

    returns:
    Plotly Go object.
    """
    data_long = df.melt(
        id_vars=[partition_col],
        value_vars=melt_cols,
        variable_name="Variable",
        value_name=value_name,
    )

    partitioned_data_long = data_long.partition_by(partition_col)
    
    if row_ordering is not None:
        row_order = dict(zip(row_ordering, range(len(row_ordering))))
        print(row_order)
        partitioned_data_long = [part.with_columns(
                                        pl.col('Variable').map_dict(row_order).alias('order_col'))
                                    .sort('order_col')
                                for part in partitioned_data_long]

    colors = n_colors(
        "rgb(0, 0, 255)", "rgb(255, 0, 0)", len(partitioned_data_long), colortype="rgb"
    )

    if color_ordering is not None:
        color_map = {ele: colors[i] for i, ele in enumerate(color_ordering)}
        curr_ordering = {
            part[partition_col][0]: i for i, part in enumerate(partitioned_data_long)
        }
        # Reorder partitioned_data_long to match color_ordering
        partitioned_data_long = [
            partitioned_data_long[curr_ordering[ele]] for ele in color_ordering
        ]
    else:
        color_map = {
            ele: colors[i]
            for i, ele in enumerate(
                data_long[partition_col].unique().to_numpy().squeeze()
            )
        }

    fig = go.Figure()

    for i, partition in enumerate(partitioned_data_long):
        x = partition[value_name].to_numpy().squeeze()
        y = partition["Variable"].to_numpy().squeeze()
        partition_name = partition[partition_col][0]
        fig.add_trace(
            go.Violin(
                x=x,
                y=y,
                legendgroup=partition_name,
                scalegroup=partition_name,
                name=partition_name,
                side="positive",
                marker_color=color_map[partition_name],
            )
        )

    fig.update_traces(
        orientation="h", side="positive", width=1.5, points=False, meanline_visible=True
    )
    fig.update_layout(
        xaxis_showgrid=False, xaxis_zeroline=False,
    )

    return fig


@polarify_in
def WandB_line_series(df: pl.DataFrame, x_axis=[], lines=[], title=""):
    """
    Plots a line series of the columns in lines, with the x-axis being the columns in x_axis.
    Logs into wandb and plots the resulting figure.
    parameters:
    df: polars dataframe.
    x_axis: (list of strings) columns to plot on the x-axis.
    lines: (list of strings) columns to plot as lines.
    title: (string) title of the plot.
    
    returns:
    None
    """
    assert wandb.run is not None, "Must be logged into wandb to use this function."
    xs = df.get_column(x_axis).to_list()
    ys = [df.get_column(col).to_list() for col in lines]
    
    wandb.log({title: wandb.plot.line_series(xs=xs, ys=ys, keys=lines, title=title)})