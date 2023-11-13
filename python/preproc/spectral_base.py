"""
This function contains base functionalities for preprocessing data in spectral domain. (Previously preproc/time_domain_features)
"""
import itertools
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from scipy import stats, signal
from sklearn.linear_model import LinearRegression


def identity(X):
    """
    Identity function. Returns input.
    Helpful if no feature engineering is desired.
    """
    return X

# TODO: split this function into two and move some of them to the transform?
def single_variate_feature_extraction(
    vec: np.ndarray,
    sampling_frequency: int,
    window_size: int,
    noverlap: int,
    band_ranges={
        "Delta": [0.5, 4],
        "Theta": [4, 8],
        "Alpha": [8, 12],
        "Beta": [12, 30],
        "Gamma": [30, 100],
    },
    additional_features: dict = {},
):
    """Extracts custom features from a vector of time series data.

    TODO: Extend to accept multiple timeseries in matrix format. Currently only works on a vector of data.

    Args:
        vec (np.array): Vector of time series data
        sampling_frequency (int): Sampling frequency of the time series data
        window_size (int): Window size to use for PSD calculation
        noverlap (int): Number of samples to overlap between windows
        band_ranges (dict): Dictionary of powerband ranges (in Hz) to use for powerband feature calculations
        additional_features (dict, optional): Additional features to extract. Defaults to {}. Should be dict of {feature_name (str): feature_function (callable))}

    Returns:
        np.array: Vector of extracted features
    """

    features = OrderedDict()

    # Time domain features
    features["mean"] = np.mean(vec)
    features["std"] = np.std(vec)
    features["min"] = np.min(vec)
    features["max"] = np.max(vec)
    features["range"] = features["max"] - features["min"]
    features["median"] = np.median(vec)
    features["samples_above"] = np.sum(vec > features["mean"] + features["std"])
    features["samples_below"] = np.sum(vec < features["mean"] - features["std"])
    features["kurtosis"] = stats.kurtosis(vec)
    features["skew"] = stats.skew(vec)
    # Auto-correlation takes a long time to calculate... so is omitted
    # features['peak_auto_corr'] = np.max(np.correlate(vec, vec, mode='full'))

    # TODO: think about combining this with function below?
    # Frequency domain features
    f, pxx = signal.welch(
        vec, fs=sampling_frequency, nperseg=window_size, noverlap=noverlap, axis=-1
    )
    pxx = np.log(pxx)

    # Get powerbands
    features |= {
        key: np.sum(pxx[np.where((f >= pb[0]) & (f < pb[1] + 0.1))[0]], axis=-1)
        for key, pb in band_ranges.items()
    }

    # Get powerband ratios
    features |= {
        f"{pb_combo[0]}/{pb_combo[1]}": features[pb_combo[0]] / features[pb_combo[1]]
        for pb_combo in itertools.combinations(band_ranges.keys(), 2)
    }

    # Slope of linear regression of PSD
    model = LinearRegression().fit(np.arange(len(vec)).reshape(-1, 1), vec)
    features["spectral_slope"] = model.coef_[0]

    # Need to convert to probability distribution?
    features["spectral_entropy"] = stats.entropy(pxx)
    features["spectral_centroid"] = np.sum(f * np.abs(pxx)) / np.sum(np.abs(pxx))

    if additional_features:
        features |= OrderedDict(
            {
                feature_name: feature_function(vec)
                for feature_name, feature_function in additional_features.items()
            }
        )

    return np.array(list(features.values()))


def get_single_variate_feature_names(additional_features: dict = {}):
    """
    Names of features returned by single_variate_feature_extraction

    Returns:
        list: List of feature names (strings)
    """
    band_ranges = {
        "Delta": [0.5, 4],
        "Theta": [4, 8],
        "Alpha": [8, 12],
        "Beta": [12, 30],
        "Gamma": [30, 100],
    }
    return (
        [
            "mean",
            "std",
            "min",
            "max",
            "range",
            "median",
            "samples_above",
            "samples_below",
            "kurtosis",
            "skew",
        ]
        + list(band_ranges.keys())
        + [
            f"{pb_combo[0]}/{pb_combo[1]}"
            for pb_combo in itertools.combinations(band_ranges.keys(), 2)
        ]
        + ["spectral_slope", "spectral_entropy", "spectral_centroid"]
        + list(additional_features.keys())
    )


def broadcast_feature_extraction_on_matrix(
    X: np.ndarray,
    sampling_frequency: int,
    window_size: int,
    noverlap: int,
    band_ranges={
        "Delta": [0.5, 4],
        "Theta": [4, 8],
        "Alpha": [8, 12],
        "Beta": [12, 30],
        "Gamma": [30, 100],
    },
    additional_features: dict = {},
):
    """Extracts custom features from a matrix of time series data. Each row should be an array of time series observations"""
    feat_extract = np.vectorize(
        single_variate_feature_extraction, signature="(n),(),(),(),(),()->(m)"
    )
    features = feat_extract(
        X,
        sampling_frequency,
        window_size,
        noverlap,
        band_ranges=band_ranges,
        additional_features=additional_features,
    )
    return features


def get_psd(
    X,
    freq_range=[0.5, 100],
    sampling_frequency=500,
    window_size=1024,
    noverlap=512,
    log=True,
):
    """
    Calculate the power spectral density of a matrix of time series data.
    Each row should be an array of time series observations.
    """
    f, pxx = signal.welch(
        X, fs=sampling_frequency, nperseg=window_size, noverlap=noverlap, axis=-1
    )
    inds = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
    pxx = pxx[:, inds]
    if log:
        pxx = np.log(pxx)
    return pxx
