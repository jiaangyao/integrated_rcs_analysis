from utils.pipeline_utils import *
import numpy as np
from sklearn.model_selection import train_test_split
# import albumentations as album
# import audiomentations as audio
from augmentation_and_correction.data_augmentation import UniversalCompose

"""
NOTE: Formatting here differs from other sub-pipelines. This is because the data augmentation pipeline is built to model the structure of libraries
Albumentation (https://albumentations.ai/docs/examples/example/) and Audiomentation (https://github.com/iver56/audiomentations).

In short, these libraries create a pipeline of transformations, where each transformation is an instance of a class with a __call__ method.

The goal is to create a super-stucture that can handle both image and audio data, and can be easily extended to include new libraries or custom transformations,
as long as they adhere to the general structure of the libraries mentioned above.
"""

import augmentation_and_correction.data_augmentation as dac
POTENTIAL_AUGMENTATION_LIBRARIES = [
    dac,
    # album,
    # audio,
]


def run_data_augmentation(X_train, y_train, groups_train, augment_conf, logger):
    
    aug_dtype_conf = augment_conf.get("data_type")
    augment_funcs= aug_dtype_conf.get("functions")
    np.random.seed(augment_conf.get("random_seed"))
    
    # First, collect all the augmentation functions and relevant parameters
    augmentation_steps = zip(
        [
            convert_string_to_callable(POTENTIAL_AUGMENTATION_LIBRARIES, func)
            for func in augment_funcs.keys()
        ],
        [(values) for values in augment_funcs.values()],
    )
    
    logger.info(f"Running data augmentation with the following steps: {augment_funcs.keys()}")
    
    # Create a pipeline with desired mix of predefined and custom transformations
    transform_pipeline = UniversalCompose([pipestep[0](*pipestep[1]) for pipestep in augmentation_steps])
    # Specifying the data type of the transform pipeline helps execute the pipeline
    if dtype := aug_dtype_conf.get("data_type") is not None:
        if dtype == "image":
            transform_pipeline = {dtype: transform_pipeline}
        elif dtype == "signal":
            transform_pipeline = {dtype: transform_pipeline, "sample_rate": aug_dtype_conf.get("sample_rate")}
    
    # Apply the pipeline to the training data
    X_train_aug = transform_pipeline(X_train)
    
    # Concatenate the augmented data and add it to the original training data
    X_train = np.concatenate([X_train, X_train_aug], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)
    groups_train = np.concatenate([groups_train, groups_train], axis=0)

    # # Collect augmented data for each augmentation function as lists
    # X_augmented = []
    # y_augmented = []
    # groups_augmented = []
    # for i, pipe_step in enumerate(augmentation_pipe):
    #     assert (
    #         pipe_step[0] is not None
    #     ), f"Function {augment_funcs[i]} not found. Verify module is imported and available in POTENTIAL_FEATURE_LIBRARIES within pipeline.subpipelines.preproce_pipeline.py."
    #     logger.info(
    #         f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}"
    #     )
    #     kwargs = pipe_step[1] if pipe_step[1] else {}
    #     X_aug, inds_augmented = pipe_step[0](X_train, **kwargs)
    #     X_augmented.append(X_aug)
    #     y_augmented.append(y_train[inds_augmented]) # Keep track of the original indices of the augmented data for the labels
    #     groups_augmented.append(groups_augmented[inds_augmented]) # Keep track of the original indices of the augmented data for the groups
    
    # # Concatenate the augmented data and add it to the training data
    # X_train = np.concatenate([X_train] + X_augmented, axis=0)
    # y_train = np.concatenate([y_train] + y_augmented, axis=0)
    # groups_train = np.concatenate([groups_train] + groups_augmented, axis=0)

    return X_train, y_train, groups_train