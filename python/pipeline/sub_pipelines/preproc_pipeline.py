"""
This function contains general wrapper functions for preprocessing data.
"""
from utils.pipeline_utils import *
import numpy as np
import numpy as np
import polars as pl


# Stand-in variable for custom time domain processing functions
import preproc.time_domain_base as tdb
import preproc.spectral_base as sb
from utils.polars_utils import extract_polars_column_as_ndarray

# Libraries for preprocessing and feature selection
import sklearn.preprocessing as skpp
import sklearn.decomposition as skd
import sklearn.feature_selection as skfs
import imblearn.over_sampling as imos
import imblearn.under_sampling as imus
import scipy.signal as scisig
import scipy.stats as scistats

# Global Variables
POTENTIAL_FEATURE_LIBRARIES = [
    tdb,
    sb,
    np,
    skpp,
    skd,
    skfs,
    imos,
    imus,
    scisig,
    scistats,
]

# TODO: Update with most recent preprocessing subpipeline


def preprocess_dataframe(data_df, preproc_funcs, logger):
    """
    Preprocesses the given DataFrame using the specified preprocessing configuration.

    Parameters:
    data_df (polars DataFrame): The data frame to be preprocessed.
    config_preprocessing (dict): Preprocessing configuration from config["preprocessing"].
    logger (Logger): Logger for logging information.
    POTENTIAL_FEATURE_LIBRARIES (dict): Dictionary of potential feature libraries.

    Returns:
    DataFrame: The preprocessed data frame.
    """
    preproc_pipe = zip(
        [
            convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func)
            for func in preproc_funcs.keys()
        ],
        [(values) for values in preproc_funcs.values()],
    )

    for pipe_step in preproc_pipe:
        logger.info(
            f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}"
        )
        kwargs = pipe_step[1] if pipe_step[1] else {}
        data_df = data_df.pipe(pipe_step[0], **kwargs)

    return data_df


def dataframe_to_matrix_export(data_df, channel_options, feature_columns, logger):
    """
    Converts a DataFrame into a numpy array with specified channel options.

    Parameters:
    data_df (DataFrame): The data frame to be converted.
    channel_options (dict): Options for channel manipulation from preproc["channel_options"].
    feature_columns (list): List of feature columns to be used in the conversion.
    logger (Logger): Logger for logging information.

    Returns:
    np.ndarray: The resulting numpy array after applying channel options.
    """

    # Convert to numpy, with desired dimensionality
    # m is num channels (i.e. num columns extracted from dataframe),
    # n is num rows (i.e. individual data measurements),
    # f is num features per row
    if channel_options["stack_channels"]:  # Stack channels into m x n x f tensor
        logger.info(f"Stacking Channels")
        X = np.stack(
            [extract_polars_column_as_ndarray(data_df, col) for col in feature_columns],
            axis=0,
        )
    elif channel_options["group_channel_rows"]:  # Group channels into n x m x f tensor
        logger.info(f"Grouping Channel Rows")
        X = np.stack(
            [extract_polars_column_as_ndarray(data_df, col) for col in feature_columns],
            axis=1,
        )
    elif channel_options[
        "concatenate_channel_rows"
    ]:  # Concatenate channels into n x (m * f) tensor
        logger.info(f"Concatenating Channel Rows")
        X = np.concatenate(
            [extract_polars_column_as_ndarray(data_df, col) for col in feature_columns],
            axis=-1,
        )
    else:
        logger.info(f"Using default column selection")
        X = data_df.select(feature_columns).to_numpy()

    logger.info(f"Feature matrix shape after preprocessing: {X.shape}")
    return X


def process_labels(data_df, label_config, logger):
    """
    Processes labels in a DataFrame according to specified label options.

    Parameters:
    data_df (DataFrame): The data frame whose labels are to be processed.
    label_config (dict): Label configuration from preproc["label_options"].
    logger (Logger): Logger for logging information.

    Returns:
    tuple: A tuple containing processed labels (y), a boolean indicating if one-hot encoding was applied,
        and group labels for cross-validation (groups), if applicable.
    """

    # Manage labels
    if label_config["label_remapping"]:
        logger.info(f"Remapping labels with {label_config['label_remapping']}")
        data_df = data_df.with_columns(
            pl.col(label_config["label_column"]).map_dict(
                label_config["label_remapping"]
            )
        )
    y = data_df.get_column(label_config["label_column"]).to_numpy().squeeze()

    if label_config["OneHotEncode"]:
        # TODO: Save y before one-hot encoding for later, save both to data object ?
        y = skpp.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        one_hot_encoded = True
    else:
        one_hot_encoded = False

    # Extract group labels for LeaveOneGroupOut cross validation (if desired)
    if label_config["group_column"]:
        logger.info(f"Using group labels for cross validation")
        groups = data_df.get_column(label_config["group_column"]).to_numpy().squeeze()
        logger.info(f"Unique groups: {np.unique(groups)}")
    else:
        groups = None

    return y, one_hot_encoded, groups


def get_features_and_labels(
    data_df, channel_options, feature_columns, label_config, logger
):
    """
    Gets the features and labels from the data frame.

    Parameters:
    data_df (DataFrame): The data frame to be used for feature and label extraction.
    config_preprocessing (dict): Preprocessing configuration from config["preprocessing"].
    logger (Logger): Logger for logging information.

    Returns:
    tuple: A tuple containing the feature matrix (X), the labels (y), a boolean indicating if one-hot encoding was applied,
        and group labels for cross-validation (groups), if applicable.
    """

    # Get features and labels
    X = dataframe_to_matrix_export(data_df, channel_options, feature_columns, logger)
    y, one_hot_encoded, groups = process_labels(data_df, label_config, logger)

    return X, y, one_hot_encoded, groups
