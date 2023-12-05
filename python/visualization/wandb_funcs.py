from collections import OrderedDict
import polars as pl
from utils.decorators import polarify_in
import wandb


@polarify_in
def WandB_line_plots(df: pl.DataFrame, x_axis="", lines=[], title=""):
    """
    Plots a .line_series of each vector in 'lines', or .line is only a single dependent variable is passed, (e.g. len(lines) == 1))
    with the x-axis being the columns in x_axis.
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
    
    if len(ys) > 1:
        wandb.log({title: wandb.plot.line_series(xs=xs, ys=ys, keys=lines, 
                    xname=x_axis, title=title)})
    else:
        data = [[x, y] for (x, y) in zip(xs, ys[0])]
        table = wandb.Table(data=data, columns = [x_axis, lines[0]])
        wandb.log({title: wandb.plot.line(table, x_axis, lines[0], title=title)})