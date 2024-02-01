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
                X = X.transpose(1, 0, 2)
            elif channel_dim >= 2:
                raise ValueError(
                    f"Channel dimension {channel_dim} is not supported for class imbalance correction. Max dimension is 1."
                )

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

            # Transpose the feature matrix back to (num_rows, num_channels, num_cols), if channel dimension is 1
            if channel_dim == 1:
                X = X.transpose(1, 0, 2)

        else:
            X, y_tmp = imb_strategy.fit_resample(X, y)
            if groups is not None:
                if groups.ndim == 1:
                    groups = groups.reshape(-1, 1)
                groups, y_group = imb_strategy.fit_resample(groups, y)
                assert np.array_equal(
                    y_tmp, y_group
                ), "Imbalance applied to group labels did not match data labels."

            y = y_tmp

        logger.info(f"Feature matrix shape after class imbalance correction: {X.shape}")
        logger.info(f"Label vector shape: {y.shape}")
    except ValueError as e:
        logger.warning(f"Class imbalance strategy error: \n\t {e}.")

    if groups is not None:
        groups = groups.squeeze()

    return X, y, groups
