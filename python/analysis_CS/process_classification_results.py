import wandb
import polars as pl
from viz_funcs import *
import altair as alt
import polars.selectors as cs
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'

def process_and_plot_confusion_matrices(conf_mat):
    
    if isinstance(conf_mat, list):
        conf_mat = np.stack(conf_mat, axis=0)
    
    labels={'x': 'Predicted Class', 'y': 'True Class'}
    # Plot average confusion matrix across folds
    title = 'Average Across Folds'
    matrix = np.mean(conf_mat, axis=0)
    mat_plot = px.imshow(matrix, title=title, labels=labels, text_auto=True)
    wandb.log({f'Confusion Matrices/{title}': mat_plot})
    
    # Plot each confusion matrix individually
    for idx, matrix in enumerate(conf_mat):
        title = f'Fold {idx + 1}'
        mat_plot = px.imshow(matrix, title=title, labels=labels, text_auto=True)
        wandb.log({f'Confusion Matrices/{title}': mat_plot})
    
    
    
    
    # Create a dataframe to hold all confusion matrices with an identifier
    # all_matrices = pd.DataFrame()
    # for idx, matrix in enumerate(conf_mat_df):
    #     df = pd.DataFrame(matrix, columns=['Predicted_0', 'Predicted_1'])
    #     df['Actual'] = df.index
    #     df['Matrix_Number'] = f'Matrix {idx + 1}'
    #     all_matrices = all_matrices.append(df, ignore_index=True)
    
    # # Melt the dataframe to long format suitable for Altair
    # all_matrices_long = all_matrices.melt(id_vars=['Actual', 'Matrix_Number'], var_name='Predicted', value_name='Count')

    # # Create an Altair chart
    # chart = alt.Chart(all_matrices_long).mark_rect().encode(
    #     x='Predicted:N',
    #     y='Actual:N',
    #     color=alt.Color('Count:Q', scale=alt.Scale(scheme='greenblue'), legend=alt.Legend(title="Count")),
    #     tooltip=['Matrix_Number', 'Actual', 'Predicted', 'Count:Q'],
    #     facet=alt.Facet('Matrix_Number:N', columns=1)
    # ).properties(
    #     width=200,
    #     height=200
    # ).interactive()

    # return chart


def process_and_log_scalar_scores(scores_df, run_dir):
    wandb.log({'Scores': wandb.Table(dataframe=scores_df.to_pandas())})
    scores_df.write_parquet(Path(f'{run_dir}/scores_by_fold.parquet'))
    
    # Process scores individually for logging.
    # First, log all scalar scores across folds as mean and std
    scores_agg = scores_df.drop('Fold').select([
        cs.numeric().mean().suffix('_mean'),
        cs.numeric().std().suffix('_std')
    ])
    wandb.log(scores_agg.to_dict(as_series=False))
    scores_agg.write_parquet(Path(f'{run_dir}/scores_agg.parquet'))
    return None


def WandB_log_plot_html(plot_html, plot_title):
    # Create a table
    table = wandb.Table(columns = [plot_title])
    # Add figure as HTML file into Table
    table.add_data(wandb.Html(plot_html))
    wandb.log({plot_title: table})
    return None


def process_and_plot_losses(scores_df, run_dir):
    # TODO: Implement Epoch Validation for early stopping...
    folds = [f'Fold {i}' for i in range(scores_df.shape[0])]
    # Average losses of each Epoch across folds
    loss_df = scores_df.select('Epoch_Losses').transpose(column_names=folds).explode(folds).with_row_count(name='Epoch')
    losses_df_agg = loss_df.select([pl.col('Epoch'),
                                    (pl.sum_horizontal(pl.col('^Fold.*$')) / len(folds)).alias('Average Loss') 
                            ])
    # Log averages to WandB
    WandB_line_plots(losses_df_agg, x_axis='Epoch', lines=['Average Loss'], title='Average Epoch Losses Across Folds')
    
    # Create plot of each individual fold, save locally then log to WandB
    loss_df = loss_df.melt(id_vars=['Epoch'], value_vars=folds, value_name='Loss', variable_name='Fold') # Data needs to be in long format for Plotly
    loss_plot = px.line(loss_df, x='Epoch', y='Loss', color='Fold')
    wandb.log({'Epoch Losses': loss_plot})
    
    # TODO: Below code does not work, logging HTML throws error.
    # TODO: Need to figure that out if we want to Altair plots.
    # # Write Plotly figure to HTML
    # # Set auto_play to False prevents animated Plotly charts 
    # # from playing in the table automatically
    # loss_html_path = Path(f'{run_dir}/loss_figure.html')
    # loss_plot.write_html(loss_html_path, auto_play = False)
    # WandB_log_plot(loss_html_path, 'Epoch Losses')
    return None


def process_and_plot_pr_curve(scores_df, run_dir):
    raise NotImplementedError
    

def process_and_log_eval_results_torch(scores, run_dir, losses=[], epoch_val=[]):
    
    # Drop prefixes (if necessary)
    scores = {
            (k.split("_", 1)[1:] if 'test_' in k else k): v
            for k, v in scores.items()
    }
    
    # Polars doesn't like multi-dimensional arrays as row elements,
    # Handle confusion matrices separately
    if 'confusion_matrix' in scores.keys():
        conf_mat = scores.pop('confusion_matrix')
        if isinstance(conf_mat, list):
            conf_mat = np.stack(conf_mat, axis=0)
    else:
        conf_mat = None
    
    # Save all scores as a table, save locally and on wandb
    scores_df = pl.DataFrame(scores).with_row_count(name='Fold')
    if losses:
        scores_df = scores_df.with_columns(pl.Series('Epoch_Losses', losses))
        
    # First, log all scalar scores across folds
    process_and_log_scalar_scores(scores_df, run_dir)
    
    # Second, log results that are not scalar scores
    # Line series visualizations: Precision-Recall, Losses
    # Log averages to WandB, and then plot all folds individually
    if losses:
        process_and_plot_losses(scores_df, run_dir)
    
    # Precision-Recall
    if 'precision_recall_curve' in scores_df.columns:
        process_and_plot_pr_curve(scores_df)
    
    # Last, log confusion matrices
    if conf_mat is not None:
        process_and_plot_confusion_matrices(conf_mat)


def process_and_log_eval_results_sklearn(scores, run_dir, losses=[]):
    pass
