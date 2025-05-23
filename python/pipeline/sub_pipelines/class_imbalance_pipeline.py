import imblearn.over_sampling as imos
import imblearn.under_sampling as imus

from utils.pipeline_utils import *

import numpy as np

POTENTIAL_IMBALANCE_LIBRARIES = [
    imos,
    imus,
]


def get_imbalance_strategy_as_object(imb_strategy_conf):
    """
    Convert imbalance strategy configuration to an imbalance object.

    Args:
        imb_config (dict): Configuration for the imbalance strategy.
            key: Class name of imbalance strategy (e.g. RandomOverSampler or imblearn.over_sampling.SMOTE)
            value: kwargs for imbalance strategy class

    Returns:
        object: Imbalance object with the specified configuration.
    """

    # Get the name of the imbalance strategy
    imb_name = list(imb_strategy_conf.keys())[0]
    # Get the kwargs for the imbalance strategy
    imb_kwargs = imb_strategy_conf[imb_name]

    # Convert the name to a callable object
    imb_obj = convert_string_to_callable(POTENTIAL_IMBALANCE_LIBRARIES, imb_name)

    # Return the imbalance object with the kwargs
    return imb_obj(**imb_kwargs)


def run_class_imbalance_correction(X, y, groups, imb_config, logger):
    """
    Applies a class imbalance correction strategy to the feature matrix and label vector.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The label vector.
        imb_config (dict): Configuration for the class imbalance strategy.
        logger: The logger object for logging information.

    Returns:
        tuple: A tuple containing the corrected feature matrix and label vector.
    """
    imb_strategy = get_imbalance_strategy_as_object(imb_config["strategy"])
    try:
        logger.info(
            f"Applying class imbalance strategy: {imb_strategy} with kwargs: {imb_strategy.get_params()}"
        )
        # Note: Assuming that the class imbalance strategy uses fit_resample method, which is the case for imblearn
        if imb_config.get("channel_options"):
            # Run class imbalance correction on each channel
            # If the channel dimension is 1, then transpose the feature matrix to be (num_channels, num_rows, num_cols)
            channel_dim = imb_config["channel_options"]["channel_dim"]
            logger.info(
                f"Channel dimension for class imbalance correction: {channel_dim}, applying class imbalance along channels."
            )
            if channel_dim == 1:
                transpose_order = (1, 0) + tuple(range(2, X.ndim))
                X = X.transpose(transpose_order)
                temp_flattened = False
            else:
                logger.warning(f"Channel dimension {channel_dim} is not debugged for class imbalance correction... Please verify appropriate class corrections.")
                transpose_order = np.arange(X.ndim)
                transpose_order[0] = channel_dim
                transpose_order[channel_dim] = 0
                X = X.transpose(transpose_order)
                temp_flattened = False
                
            if X.ndim >= 3: # Imblearn only likes 2D arrays.. need to flatten extra dimensions
                temp_flattened = True
                # Store the original shape
                original_shape = X.shape
                # Flatten the array after the 2nd dimension
                # Calculate the new shape: the product of dimensions from the 3rd dimension onwards
                new_shape = X.shape[:2] + (-1,)
                X = X.reshape(new_shape)
            else: 
                temp_flattened = False
                
            X_list = []
            groups_list = []
            y_list = []
            y_group = []
            for i in range(X.shape[0]):
                X_tmp, y_tmp = imb_strategy.fit_resample(X[i], y)
                X_list.append(X_tmp)
                if groups is not None:
                    if groups.ndim == 1:
                        groups = groups.reshape(-1, 1)
                    groups_tmp, y_group = imb_strategy.fit_resample(groups, y)
                    groups_list.append(groups_tmp)
                    y_group.append(y_group)
                y_list.append(y_tmp)

            X = np.stack(X_list, axis=0)
            if len(y_list) > 1:
                for i in range(0, len(y_list) - 1):
                    assert np.array_equal(
                        y_list[i], y_list[i + 1]
                    ), "Label vectors are not equal after class imbalance correction."
                    if groups is not None:
                        assert np.array_equal(
                            groups_list[i], groups_list[i + 1]
                        ), "Group vectors are not equal after class imbalance correction."
                        assert np.array_equal(
                            y[i], y_group[i]
                        ), "Imbalance applied to group labels did not match data labels."

            y = y_list[0]
            if groups:
                groups = groups_list[0]

            if temp_flattened: # Get back to the original feature shape (with channels as 0th dimension)
                new_shape = X.shape[:2] + original_shape[2:]
                X = X.reshape(new_shape)
            
            # Transpose the feature matrix back to (num_rows, num_channels, num_cols), if channel dimension is 1
            if channel_dim == 1:
                X = X.transpose(transpose_order)
            elif channel_dim > 1:
                X = X.transpose(transpose_order)
            

        else: # Run class imbalance correction on the entire feature matrix, not by channel
            
            if X.ndim >= 3: # Imblearn only likes 2D arrays.. need to temporarily flatten extra dimensions
                temp_flattened = True
                # Store the original shape
                original_shape = X.shape
                # Flatten the array after the 2nd dimension
                # Calculate the new shape: the product of dimensions from the 3rd dimension onwards
                new_shape = [X.shape[0]] + [-1]
                X = X.reshape(new_shape)
            else:
                temp_flattened = False
                
            X, y_tmp = imb_strategy.fit_resample(X, y)
            if groups is not None:
                groups_mapping = dict(zip(np.unique(groups), np.arange(len(np.unique(groups)))))
                groups_mapped = np.array([groups_mapping[g] for g in groups])
                if groups_mapped.ndim == 1:
                    groups_mapped = groups_mapped.reshape(-1, 1)
                groups_mapped, y_group = imb_strategy.fit_resample(groups_mapped, y)
                groups_unmapping = {v: k for k, v in groups_mapping.items()}
                groups = np.array([groups_unmapping[g] for g in groups_mapped.flatten()])
                assert np.array_equal(
                    y_tmp, y_group
                ), "Imbalance applied to group labels did not match data labels."
            
            if temp_flattened:
                new_shape = (X.shape[0],) + original_shape[1:]
                X = X.reshape(new_shape)

            y = y_tmp

        # logger.info(f"Feature matrix shape after class imbalance correction: {X.shape}")
        # logger.info(f"Label vector shape: {y.shape}")
    except ValueError as e:
        logger.warning(f"Class imbalance strategy error: \n\t {e}.")

    if groups is not None:
        groups = groups.squeeze()

    return X, y, groups
