"""
This function contains base functionalities for preprocessing data in spectral domain. (Previously preproc/time_domain_features)
"""
import itertools
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from scipy import stats, signal
from sklearn.linear_model import LinearRegression
import polars as pl


def multivariate_feature_embedding(matrix: npt.NDArray[np.float64], window_size: int, noverlap: int, sampling_frequency: int, band_range={
        "Delta": [0.5, 4],
        "Theta": [4, 8],
        "Alpha": [8, 12],
        "Beta": [12, 30],
        "Gamma": [30, 60],
    }):
    """Extracts custom features from a matrix of mulitvariate time series samples.
    Slides a window across the multivariate time series and extracts features from each window.
    As in: m (num channels) x s (num samples) -> f (num features) x w (num windows)
    """
    num_windows = int(np.ceil((matrix.shape[1] - window_size) / (window_size - noverlap))) + 1
    
    for i in range(num_windows):
        window = matrix[:, i * (window_size - noverlap) : i * (window_size - noverlap) + window_size]
        single_channel_features = broadcast_feature_extraction_on_matrix(window, sampling_frequency, window_size, noverlap=0, band_ranges=band_range).reshape(-1, 1)
        cross_channel_features = np.array(list(cross_channel_coherence(window, sampling_frequency, band_ranges=band_range).values())).reshape(-1, 1)
        features = np.vstack((single_channel_features, cross_channel_features))
        if i == 0:
            feature_matrix = features
        else:
            feature_matrix = np.hstack((feature_matrix, features))
            
    if np.any(np.isinf(feature_matrix)):
        print("WARNING: Inf values in feature matrix")
        feature_matrix = np.nan_to_num(feature_matrix)
    return feature_matrix


def cross_channel_coherence(matrix, sampling_frequency, band_ranges={
        "Delta": [0.5, 4],
        "Theta": [4, 8],
        "Alpha": [8, 12],
        "Beta": [12, 30],
        "Gamma": [30, 60],
    },
):
    """
    Compute coherence in specific frequency bands for each combination of channels.

    :param matrix: 2D numpy array where each row is a signal (channel)
    :param sampling_frequency: Sampling frequency in Hz
    :param band_ranges: Dictionary with frequency bands
    :return: Dictionary with coherence values for each channel pair and frequency band
    """
    n_channels = matrix.shape[0]
    channel_pairs = itertools.combinations(range(n_channels), 2)
    band_coherence = OrderedDict()

    for (ch1, ch2) in channel_pairs:
        f, coh = signal.coherence(matrix[ch1], matrix[ch2], fs=sampling_frequency)
        
        for band_name, (f_low, f_high) in band_ranges.items():
            # Extract coherence values within the frequency band
            band_mask = (f >= f_low) & (f <= f_high)
            band_coh = coh[band_mask]
            avg_band_coh = np.mean(band_coh) if len(band_coh) > 0 else np.nan

            band_coherence[(ch1, ch2, band_name)] = avg_band_coh

    return band_coherence


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
        "Gamma": [30, 60],
    },
    additional_features: dict = {},
):
    """Extracts custom features from a vector of time series data.
    As in, converts a vector of time series data into a vector of features. 1 x n -> 1 x m

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
    features["iqr"] = np.subtract(*np.percentile(vec, [75, 25]))
    features["median"] = np.median(vec)
    features["samples_above"] = np.sum(vec > features["mean"] + features["std"])
    features["samples_below"] = np.sum(vec < features["mean"] - features["std"])
    features["kurtosis"] = stats.kurtosis(vec)
    features["skew"] = stats.skew(vec)
    # Auto-correlation takes a long time to calculate... so is omitted
    # features['peak_auto_corr'] = np.max(np.correlate(vec, vec, mode='full'))

    # TODO: think about combining this with function below?
    # Frequency domain features
    f, pxx_raw = signal.welch(
        vec, fs=sampling_frequency, nperseg=window_size, noverlap=noverlap, axis=-1
    )
    pxx = np.log(pxx_raw)

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
    features["spectral_intercept"] = model.intercept_

    # Need to convert to probability distribution?
    features["spectral_entropy"] = stats.entropy(pxx_raw)
    features["spectral_centroid"] = np.sum(f * np.abs(pxx)) / np.sum(np.abs(pxx))
    features["spectral_energy"] = np.sum(pxx)

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
        + ["spectral_slope", "spectral_intercept", "spectral_entropy", "spectral_centroid", "spectral_energy"]
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


def get_psd_polars(
    df: pl.DataFrame,
    td_columns=[],
    freq_range=[0.5, 100],
    sampling_frequency=500,
    window_size=1024,
    noverlap=512,
):
    """
    Calculate the power spectral density (PSD) for each time domain column in the DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame containing the time domain data.
        td_columns (list): List of column names in the DataFrame that contain the time domain data.
        freq_range (list, optional): The frequency range for calculating the PSD. Defaults to [0.5, 100].
        sampling_frequency (int, optional): The sampling frequency of the time domain data. Defaults to 500.
        window_size (int, optional): The size of the window used for calculating the PSD. Defaults to 1024.
        noverlap (int, optional): The number of samples to overlap between windows. Defaults to 512.
        log (bool, optional): Whether to apply logarithm to the PSD values. Defaults to True.

    Returns:
        pl.DataFrame: The DataFrame with additional columns for the PSD values and frequency values.
    """
    
    # Assumes each time domain column is already epoched (e.g. time_domain_base.epoch_data was called)
    for col in td_columns:
        # Calculate PSD
        f, pxx = signal.welch(
            df.get_column(col).to_numpy(), fs=sampling_frequency, nperseg=window_size, noverlap=noverlap, axis=-1
        )
        
        # Select PSD values within desired frequency range
        pxx = pxx[:, np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]]
        
        # Add PSD values to DataFrame
        df = df.with_columns(
            pl.Series(
                name=f"{col}_psd",
                values= pxx,
                dtype=pl.Array(inner=pl.Float32, width=pxx.shape[1])
            )
        )
    
    # Add frequency values to DataFrame. Entire column will be an identical vector of frequency values.
    f_mat = np.tile(f[np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]], (df.height, 1))
    df = df.with_columns(
        pl.Series(
            name=f"psd_freq",
            values=f_mat,
            dtype=pl.Array(inner=pl.Float32, width=f_mat.shape[1])
        )
    )
    
    return df
    


def get_psd(
    X: npt.NDArray[np.float64],
    freq_ranges=[[0.5, 100]],
    sampling_frequency=500,
    window_size=1024,
    noverlap=512,
    log=True,
):
    """
    Calculate the power spectral density of a matrix of time series data.
    Each row should be an array of time series observations.
    """
    # Handle frequency ranges edge cases. Ensure freq_ranges is a 2D array where each row is a frequency range band to keep.
    if ~isinstance(freq_ranges, np.ndarray): freq_ranges = np.array(freq_ranges)
    if freq_ranges.ndim == 1: freq_ranges = freq_ranges.reshape(1, -1)
    
    f, pxx = signal.welch(
        X, fs=sampling_frequency, nperseg=window_size, noverlap=noverlap, axis=-1
    )
    
    # Concatenate PSDs for each desired frequency range into single matrix    
    pxx = np.concatenate([pxx[:, np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]] for freq_range in freq_ranges], axis=-1)
    
    if log:
        pxx = np.log(pxx)
        
    return pxx