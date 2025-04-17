from collections import OrderedDict
from utils.pipeline_utils import *
import numpy as np
from .preproc_pipeline import POTENTIAL_FEATURE_LIBRARIES

def reject_outliers_artifacts(X, y, groups, preproc_funcs, logger):
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
    reject_pipe = zip(
        [
            convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func)
            for func in preproc_funcs.keys()
        ],
        [(values) for values in preproc_funcs.values()],
    )

    for i, pipe_step in enumerate(reject_pipe):
        assert (
            pipe_step[0] is not None
        ), f"Function {preproc_funcs[i]} not found. Verify module is imported and available in POTENTIAL_FEATURE_LIBRARIES within pipeline.subpipelines.preproce_pipeline.py."
        logger.info(
            f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}"
        )
        kwargs = pipe_step[1] if pipe_step[1] else {}
        # data_df = data_df.pipe(pipe_step[0], **kwargs)
        X, y, groups = pipe_step[0](X, y, groups, **kwargs)

    return X, y, groups

