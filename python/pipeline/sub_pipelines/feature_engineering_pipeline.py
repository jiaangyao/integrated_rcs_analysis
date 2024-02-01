from collections import OrderedDict
from utils.pipeline_utils import *
import numpy as np
from .preproc_pipeline import POTENTIAL_FEATURE_LIBRARIES


def create_feature_pipe_object(feature_eng_funcs):
    """
    Creates a feature transformation pipeline object based on the given feature engineering configuration.

    Parameters:
    feature_eng_config (dict): Feature engineering configuration from config["feature_engineering"].

    Returns:
    tuple: A tuple containing the feature pipeline object and a string representation of the pipeline steps.
    """
    function_calls = tuple(
        zip(
            [
                convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func)
                for func in feature_eng_funcs.keys()
            ],
            [(values) for values in feature_eng_funcs.values()],
        )
    )
    feature_steps = OrderedDict(zip(feature_eng_funcs.keys(), function_calls))

    feature_pipe, pipe_step_names = create_transform_pipeline(feature_steps)
    pipe_string = "\n".join([f"{step}" for step in pipe_step_names])

    return feature_pipe, pipe_string


def process_features(
    X,
    feature_pipe,
    pipe_string,
    channel_options,
    stacked_channels,
    num_channels,
    num_rows,
    logger,
):
    """
    Applies a feature extraction pipeline to the data and transforms it based on channel options.

    Parameters:
    X (np.ndarray): The input data on which feature extraction is to be applied.
    channel_options (dict): Options for channel manipulation in final data matrix. Note, these are different from the channel options in the preprocessing pipeline.
    feature_pipe (Pipeline): The feature extraction pipeline.
    pipe_string (str): A string representation of the pipeline steps.
    logger (Logger): Logger for logging information.

    Returns:
    np.ndarray: The transformed data after applying feature extraction and channel operations.
    """
    # Apply feature extraction pipeline on data
    logger.info(f"Transforming data into features with pipeline: \n {pipe_string}")

    # Check if user wants to apply feature pipeline by channel, or specified dimension
    if channel_options["pipe_on_dim_0"]:
        X = np.stack(
            [feature_pipe.fit_transform(X[i]) for i in range(X.shape[0])], axis=0
        )
    else:
        X = feature_pipe.fit_transform(X)

    # Check if user wants to aggregate features across channels (which is likely dim=0)
    if channel_options["concat_channel_features"]:
        if stacked_channels:
            # If channels are stacked, then we need to transpose the channel and row dimensions, then reshape to concatenate features across channels
            X = np.transpose(X, (1, 0, 2))
        X = np.reshape(X, (X.shape[0], -1))
    elif channel_options["average_channel_features"]:
        X = np.mean(X, axis=0)
    elif channel_options["group_channel_features_by_row_after_pipe"]:
        X = np.transpose(X, (1, 0, 2))
    elif channel_options["stack_channel_features"]:
        # ! NOTE: THIS IS NOT DEBUGGED YET, should not be used right now...
        X = np.reshape(X, (len(num_channels), num_rows, -1))

    return X
