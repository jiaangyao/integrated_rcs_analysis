from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import zscore

def principle_component_analysis(data, **kwargs):
    """
    Perform Principle Component Analysis on the input data.

    Parameters:
    data (DataFrame): The input data to transform.

    Returns:
    transformed_data (DataFrame): The transformed data after PCA.
    """
    pca = PCA(**kwargs)
    transformed_data = pca.fit_transform(data)
    return transformed_data

def zscore_by_group(X, y, groups, axis=0):
    """
    Z-scores the values in X along the specified axis for each group in groups.
    
    Parameters:
    - X: np.ndarray, the data to be z-scored
    - y: np.ndarray, the target values
    - groups: np.ndarray, the group labels
    - axis: int, the axis along which to z-score the data
    
    Returns:
    - X_zscored: np.ndarray, the z-scored data
    """
    assert groups is not None, "Groups must be provided for z-scoring by group"
    
    unique_groups = np.unique(groups)
    X_zscored = np.zeros_like(X)
    
    for group in unique_groups:
        group_mask = (groups == group)
        X_group = X[group_mask]
        X_zscored[group_mask] = zscore(X_group, axis=axis)
    
    return X_zscored, y, groups


def first_order_log_10_estimate(data):
    """
    Estimate the log base 10 of the data with a linear function. It will esimate the log of the data around the mean of the data using a first order Taylor expansion.
    """
    mean_data = np.mean(data)
    zeroth_order = np.log10(mean_data)
    first_order = (data - mean_data) / (mean_data * np.log(10)) + zeroth_order # First term corresponds to the first derivative of the log function. Second term is the zeroth order term.
    return first_order