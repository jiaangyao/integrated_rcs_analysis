import itertools
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from scipy import stats, signal
import numpy as np


def single_variate_feature_extraction(vec: np.ndarray, sampling_frequency: int, window_size: int, noverlap: int, 
                                    band_ranges={'Delta': [0.5, 4], 'Theta': [4,8], 'Alpha': [8, 12], 'Beta': [12, 30], 'Gamma': [30, 100]},
                                    additional_features: dict = {}):
    """Extracts custom features from a vector of time series data.

    Args:
        vec (np.array): Vector of time series data
        sampling_frequency (int): Sampling frequency of the time series data
        window_size (int): Window size to use for PSD calculation
        noverlap (int): Number of samples to overlap between windows
        band_ranges (dict): Dictionary of powerband ranges (in Hz) to use for powerband feature calculations
        additional_features (dict, optional): Additional features to extract. Defaults to {}. Should be dict of {feature_name: feature_function}

    Returns:
        np.array: Vector of extracted features
    """
    
    features = OrderedDict()
    
    # Time domain features
    features['mean'] = np.mean(vec)
    features['std'] = np.std(vec)
    features['min'] = np.min(vec)
    features['max'] = np.max(vec)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(vec)
    features['samples_above'] = np.sum(vec > features['mean'] + features['std'])
    features['samples_below'] = np.sum(vec < features['mean'] - features['std'])
    features['kurtosis'] = stats.kurtosis(vec)
    features['skew'] = stats.skew(vec)
    # Auto-correlation takes a long time to calculate... so is omitted
    #features['peak_auto_corr'] = np.max(np.correlate(vec, vec, mode='full'))
    
    # Frequency domain features
    f, pxx = signal.welch(vec, fs=sampling_frequency, nperseg=window_size, noverlap=noverlap, axis=-1)
    pxx = np.log(pxx)
    
    # Get powerbands
    features |= {key: np.sum(
                        pxx[ np.where( (f >= pb[0]) & (f < pb[1]+0.1) )[0] ],
                    axis=-1)
                for key, pb in band_ranges.items()}
    
    # Get powerband ratios
    features |= {f'{pb_combo[0]}/{pb_combo[1]}': features[pb_combo[0]] / features[pb_combo[1]] for pb_combo in itertools.combinations(band_ranges.keys(), 2)}
    
    # Slope of linear regression of PSD
    model = LinearRegression().fit(np.arange(len(vec)).reshape(-1,1), vec)
    features['spectral_slope'] = model.coef_[0]
    
    # Need to convert to probability distribution?
    features['spectral_entropy'] = stats.entropy(pxx)
    features['spectral_centroid'] = np.sum(f * np.abs(pxx)) / np.sum(np.abs(pxx))
    
    if additional_features:
        features |= OrderedDict({feature_name: feature_function(vec) for feature_name, feature_function in additional_features.items()})
    
    return np.array(list(features.values()))


def get_single_variate_feature_names(additional_features: dict = {}):
    """ 
    Names of features returned by single_variate_feature_extraction

    Returns:
        list: List of feature names (strings)
    """
    band_ranges = {'Delta': [0.5, 4], 'Theta': [4,8], 'Alpha': [8, 12], 'Beta': [12, 30], 'Gamma': [30, 100]}
    return ['mean', 'std', 'min', 'max', 'range', 'median', 'samples_above', 'samples_below', 'kurtosis', 'skew'] + \
            list(band_ranges.keys()) + \
            [f'{pb_combo[0]}/{pb_combo[1]}' for pb_combo in itertools.combinations(band_ranges.keys(), 2)] + \
            ['spectral_slope', 'spectral_entropy', 'spectral_centroid'] + \
            list(additional_features.keys())