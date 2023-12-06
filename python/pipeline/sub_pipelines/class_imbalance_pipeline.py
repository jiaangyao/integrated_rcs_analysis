import imblearn.over_sampling as imos
import imblearn.under_sampling as imus

from utils.pipeline_utils import *

POTENTIAL_IMBALANCE_LIBRARIES = [
    imos,
    imus,
]

def get_imbalance_strategy_as_object(imb_config):
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
    imb_name = list(imb_config["strategy"].keys())[0]
    # Get the kwargs for the imbalance strategy
    imb_kwargs = imb_config["strategy"][imb_name]
    
    # Convert the name to a callable object
    imb_obj = convert_string_to_callable(POTENTIAL_IMBALANCE_LIBRARIES, imb_name)
    
    # Return the imbalance object with the kwargs
    return imb_obj(**imb_kwargs)


def run_class_imbalance_correction(X, y, imb_config, logger):
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
    imb_strategy = get_imbalance_strategy_as_object(imb_config)
    try:
        logger.info(f"Applying class imbalance strategy: {imb_strategy} with kwargs: {imb_strategy.get_params()}")
        # Note: Assuming that the class imbalance strategy uses fit_resample method, which is the case for imblearn
        X, y = imb_strategy.fit_resample(X, y)
        logger.info(f"Feature matrix shape after class imbalance correction: {X.shape}")
        logger.info(f"Label vector shape: {y.shape}")
    except ValueError as e:
        logger.warning(f"No class imbalance strategy applied due to error: \n\t {e}.")

    return X, y