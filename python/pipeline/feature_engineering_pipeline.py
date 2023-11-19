from collections import OrderedDict
from utils.pipeline_utils import *
import numpy as np

def create_feature_pipe_object(feature_eng_config, POTENTIAL_FEATURE_LIBRARIES, create_transform_pipeline):
    """
    Creates a feature transformation pipeline object based on the given feature engineering configuration.

    Parameters:
    feature_eng_config (dict): Feature engineering configuration from config["feature_engineering"].
    POTENTIAL_FEATURE_LIBRARIES (dict): A dictionary mapping string identifiers to actual callable functions or objects.
    create_transform_pipeline (function): A function to create the transformation pipeline.

    Returns:
    tuple: A tuple containing the feature pipeline object and a string representation of the pipeline steps.
    """
    
    feature_eng_funcs = feature_eng_config["functions"]
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


def process_features(X, channel_options, feature_pipe, pipe_string, logger):
    """
    Applies a feature extraction pipeline to the data and transforms it based on channel options.

    Parameters:
    X (np.ndarray): The input data on which feature extraction is to be applied.
    channel_options (dict): Options for channel manipulation.
    feature_pipe (Pipeline): The feature extraction pipeline.
    pipe_string (str): A string representation of the pipeline steps.
    logger (Logger): Logger for logging information.

    Returns:
    np.ndarray: The transformed data after applying feature extraction and channel operations.
    """

    logger.info(f"Transforming data into features with pipeline: \n {pipe_string}")

    # Apply feature pipeline by channel, or specified dimension
    if channel_options["pipe_on_dim_0"]:
        X = np.stack(
            [feature_pipe.fit_transform(X[i]) for i in range(X.shape[0])], axis=0
        )
    else:
        X = feature_pipe.fit_transform(X)
    
    # Aggregate features across channels
    if channel_options["concat_channel_features"]:
        X = np.reshape(X, (X.shape[0], -1))
    elif channel_options["average_channel_features"]:
        X = np.mean(X, axis=0)
    elif channel_options["group_channel_features_by_row_after_pipe"]:
        X = np.transpose(X, (1, 0, 2))
    elif channel_options["stack_channel_features"]:
        # Note: This option is not debugged yet and should not be used currently.
        raise NotImplementedError("Stack channel features option is not debugged and should not be used.")
        # X = np.reshape(X, (len(preproc['feature_columns']), data_df.height, -1))

    return X