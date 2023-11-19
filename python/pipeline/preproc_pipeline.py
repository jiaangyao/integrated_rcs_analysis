"""
This function contains general wrapper functions for preprocessing data.
"""
from utils.pipeline_utils import *
import numpy as np
import numpy as np
import polars as pl
import sklearn.preprocessing as skpp

def preprocess_dataframe(data_df, config_preprocessing, logger, POTENTIAL_FEATURE_LIBRARIES):
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

    preproc_funcs = config_preprocessing["functions"]
    preproc_pipe = zip(
        [
            convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func)
            for func in preproc_funcs.keys()
        ],
        [(values) for values in preproc_funcs.values()],
    )
    
    for pipe_step in preproc_pipe:
        logger.info(f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}")
        data_df = data_df.pipe(pipe_step[0], **pipe_step[1])

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
    if channel_options["stack_channels"]:  # Stack channels into m x n x f tensor
        logger.info("Stacking Channels")
        X = np.stack(
            [data_df.get_column(col).to_numpy() for col in feature_columns],
            axis=0,
        )
    elif channel_options["group_channel_rows"]:  # Group channels into n x m x f tensor
        logger.info("Grouping Channel Rows")
        X = np.stack(
            [data_df.get_column(col).to_numpy() for col in feature_columns],
            axis=1,
        )
    elif channel_options["concatenate_channel_rows"]:  # Concatenate channels into n x (m * f) tensor
        logger.info("Concatenating Channel Rows")
        X = np.concatenate(
            [data_df.get_column(col).to_numpy() for col in feature_columns],
            axis=-1,
        )
    else:
        raise NotImplementedError(
            "Only stacking, grouping, or concatenating channels is currently supported"
        )
    
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

    one_hot_encoded = False
    if label_config["OneHotEncode"]:
        logger.info("Applying OneHotEncoding to labels")
        y = skpp.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        one_hot_encoded = True

    groups = None
    if label_config.get("group_column"):
        logger.info("Using group labels for cross validation")
        groups = data_df.get_column(label_config["group_column"]).to_numpy().squeeze()
        logger.info(f"Unique groups: {np.unique(groups)}")

    return y, one_hot_encoded, groups


def get_features_and_labels():
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
    X = dataframe_to_matrix_export()
    y, one_hot_encoded, groups = process_labels(data_df, label_config, logger)

    return X, y, one_hot_encoded, groups