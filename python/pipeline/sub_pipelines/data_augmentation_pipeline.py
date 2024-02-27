from utils.pipeline_utils import *
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as album
import audiomentations as audio
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
    album,
    audio,
]


def run_data_augmentation(X_train, y_train, groups_train, augment_conf, logger):
    
    # First, need to verify that channel dim is first dim. If not, need to transpose
    channel_dim = augment_conf.get("channel_dim")
    if channel_dim != 0:
        transpose_order = np.arange(X_train.ndim)
        transpose_order[0] = channel_dim
        transpose_order[channel_dim] = 0
        X_train = np.transpose(X_train, transpose_order)
    np.random.seed(augment_conf.get("random_seed"))
    
    aug_dtype_conf = augment_conf.get("data_type")
    
    # Create a list to store the augmented data. Each element of the list will be the augmented data for a different grouping of augmentation functions
    X_train_aug = []
    y_train_aug = []
    groups_train_aug = []
    for i, function_group in enumerate(aug_dtype_conf.get("augment_groupings")):
        augment_funcs= function_group
        augment_funcs = {key: value if value else {} for key, value in augment_funcs.items()}
        
        # Be explicit for logging
        augmentation_steps = zip(
            [
                convert_string_to_callable(POTENTIAL_AUGMENTATION_LIBRARIES, func)
                for func in augment_funcs.keys()
            ],
            [(values) for values in augment_funcs.values()],
        )
        
        logger.info(f"Running data augmentation for augment function groups {i}.")
        [logger.info(
                f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}"
            ) 
        for pipe_step in augmentation_steps]
        
        # Create a list of instances of the augmentation functions... not great that I am repeating code from the snippet above
        augmentation_instances = [convert_string_to_callable(POTENTIAL_AUGMENTATION_LIBRARIES, func)(**values) for func, values in augment_funcs.items()]
        
        # Create a pipeline with desired mix of predefined and custom transformations
        transform_pipeline = UniversalCompose(augmentation_instances)
        
        # Apply the pipeline to the training data
        # Specifying the data type of the transform pipeline helps execute the pipeline
        if dtype := aug_dtype_conf.get("data_type"):
            if dtype == "image":
                X_train_dict = {dtype: X_train}
            elif dtype == "signal":
                X_train_dict = {dtype: X_train, "sample_rate": aug_dtype_conf.get("sample_rate")}
        
        # Run pipe
        X_train_aug_tmp = transform_pipeline(X_train_dict)
        
        # Add the augmented data to the list. 
        X_train_aug.append(X_train_aug_tmp)
        y_train_aug.append(y_train)
        if groups_train is not None:
            groups_train_aug.append(groups_train)
    
    # If the channel dimension was not the first dimension, need to transpose back
    if channel_dim != 0:
        X_train_aug = [np.transpose(X_train_aug_ele, transpose_order) for X_train_aug_ele in X_train_aug]
        X_train = np.transpose(X_train, transpose_order)
    
    # Concatenate the augmented data and add it to the original training data
    X_train = np.concatenate([X_train, *X_train_aug], axis=0)
    y_train = np.concatenate([y_train, *y_train_aug], axis=0)
    if groups_train is not None:
        groups_train = np.concatenate([groups_train, *groups_train_aug], axis=0)
        
    logger.info(f"Data augmentation complete. New training data shape: {X_train.shape}")

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