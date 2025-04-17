import numpy as np
from sklearn.neighbors import NearestNeighbors

# Outlier rejection using k-NN
def kNN_outlier_rejection(X, y, groups=None, k_neighbors=5, percentile_cutoff=95, axis=0):
    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k_neighbors)
    nbrs.fit(X)

    # Calculate distances to the k nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    # Average distance to k nearest neighbors for each point
    outlier_scores = np.mean(distances, axis=1)

    # Identify outliers (let's say the top 5% are considered outliers)
    threshold = np.percentile(outlier_scores, percentile_cutoff)
    outliers = np.where(outlier_scores > threshold)[0]
    
    # Remove outliers from X, y
    X = np.delete(X, outliers, axis=axis)
    y = np.delete(y, outliers, axis=axis)
    if groups is not None:
        groups = np.delete(groups, outliers)
    
    return X, y, groups


def remove_nan_rows(X, y, groups=None, axis=1):
    nan_rows = np.isnan(X).any(axis=axis)
    X = X[~nan_rows]
    y = y[~nan_rows]
    if groups is not None:
        groups = groups[~nan_rows]
    return X, y, groups
