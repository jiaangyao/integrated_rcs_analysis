from typing import Union
import polars as pl
from scipy.signal import butter, sosfiltfilt
import numpy.typing as npt
from scipy.signal import hilbert, decimate, resample, resample_poly
import numpy as np


TIME_SERIES_T = Union[pl.Series, pl.Expr]
FLOAT_EXPR = Union[float, pl.Expr]
FLOAT_INT_EXPR = Union[int, float, pl.Expr]
INT_EXPR = Union[int, pl.Expr]


def process_signal(data, downsample_factor, method='decimate'):
    """
    Apply a Hilbert transform to the input data and then downsample it using the specified method.

    Parameters:
    data (array_like): Input signal data.
    downsample_factor (int): Factor by which the data is downsampled.
    method (str): Method to use for downsampling. Options are 'decimate', 'resample', or 'resample_poly'.

    Returns:
    ndarray: The processed signal.
    """
    # Handle case where data has NaNs (likely due to filtering failure)
    if np.any(np.isnan(data)):
        return np.array(np.nan)
    
    # Apply the Hilbert transform
    transformed_data = np.abs(hilbert(data))

    # Downsample the data using the specified method
    if method == 'decimate':
        downsampled_data = decimate(transformed_data, downsample_factor)
    elif method == 'resample':
        downsampled_data = resample(transformed_data, int(len(data) / downsample_factor))
    elif method == 'resample_poly':
        downsampled_data = resample_poly(transformed_data, 1, downsample_factor)
    else:
        raise ValueError("Invalid downsampling method. Choose 'decimate', 'resample', or 'resample_poly'.")

    return downsampled_data


def try_filter(sos, x):
    try:
        filtered = sosfiltfilt(sos, x)
    except ValueError as e:
        # print(f'Error: {e}')
        # print(f'Length of x: {len(x)}')
        # print('Returning Null')
        filtered = np.array(np.nan)
    return filtered

def butterworth_bandpass_np(x: npt.NDArray , N, Wn, fs) -> pl.Series:
    sos = butter(N, Wn, 'bandpass', fs=fs, output='sos')
    filtered = try_filter(sos, x)
    return filtered

def butterworth_bandpass(x: TIME_SERIES_T, N: int, Wn: list, fs: int) -> TIME_SERIES_T:
    # NOTE: Using map_elements is necessary when working within an aggregation() call, 
    # otherwise, when calling on entire column, need to call map_batches() instead.
    return x.map_elements(lambda series: pl.Series(butterworth_bandpass_np(series.to_numpy(), N, Wn, fs)))


def _bandpass_envelope_downsample(x: TIME_SERIES_T, N: int, Wn: list, fs: int, downsampling: int) -> TIME_SERIES_T:
    # NOTE: Using map_elements is necessary when working within an aggregation() call, 
    # otherwise, when calling on entire column, need to call map_batches() instead.
    return x.map_elements(lambda series:
        pl.Series(
            process_signal(
                butterworth_bandpass_np(series.to_numpy(), N, Wn, fs),
                downsampling
                )
        )
    )

    
def butterworth_lowpass_np(x: npt.NDArray , N, Wn, fs) -> pl.Series:
    sos = butter(N, Wn, 'lowpass', fs=fs, output='sos')
    filtered = try_filter(sos, x)
    return pl.Series(filtered)


def butterworth_lowpass(x: TIME_SERIES_T, N: int, Wn: list, fs: int) -> TIME_SERIES_T:
    return x.map_elements(lambda series: butterworth_lowpass_np(series.to_numpy(), N, Wn, fs))


def butterworth_highpass_np(x: npt.NDArray , N, Wn, fs) -> pl.Series:
    sos = butter(N, Wn, 'highpass', fs=fs, output='sos')
    filtered = try_filter(sos, x)
    return pl.Series(filtered)

def butterworth_highpass(x: TIME_SERIES_T, N: int, Wn: list, fs: int) -> TIME_SERIES_T:
    return x.map_elements(lambda series: butterworth_highpass_np(series.to_numpy(), N, Wn, fs))


@pl.api.register_expr_namespace("filt")
class TimeDomainFiltering:
    def __init__(self, expr: pl.Expr):
        self._expr = expr
        
    def butterworth_bp(self, N, Wn, fs) -> pl.Expr:
        return butterworth_bandpass(self._expr, N, Wn, fs)
    
    def bandpass_envelope_downsample(self, N, Wn, fs, downsample) -> pl.Expr:
        return _bandpass_envelope_downsample(self._expr, N, Wn, fs, downsample)
    
    def butterworth_hp(self, N, Wn, fs) -> pl.Expr:
        return butterworth_lowpass(self._expr)
    
    def butterworth_lp(self, N, Wn, fs) -> pl.Expr:
        return butterworth_highpass(self._expr)
    
    