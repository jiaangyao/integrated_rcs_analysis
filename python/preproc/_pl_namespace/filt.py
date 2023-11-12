from typing import Union
import polars as pl
from scipy.signal import butter, sosfiltfilt
import numpy.typing as npt


TIME_SERIES_T = Union[pl.Series, pl.Expr]
FLOAT_EXPR = Union[float, pl.Expr]
FLOAT_INT_EXPR = Union[int, float, pl.Expr]
INT_EXPR = Union[int, pl.Expr]

def try_filter(sos, x):
    try:
        filtered = sosfiltfilt(sos, x)
    except ValueError as e:
        print(f'Error: {e}')
        print(f'Length of x: {len(x)}')
        print('Returning Null')
        return pl.Series(pl.lit(None))
    return filtered

def butterworth_bandpass_np(x: npt.NDArray , N, Wn, fs) -> pl.Series:
    sos = butter(N, Wn, 'bandpass', fs=fs, output='sos')
    filtered = try_filter(sos, x)
    return pl.Series(filtered)

def butterworth_bandpass(x: TIME_SERIES_T, N: int, Wn: list, fs: int) -> TIME_SERIES_T:
    # NOTE: Using map_elements is necessary when working within an aggregation() call, 
    # otherwise, when calling on entire column, need to call map_batches() instead.
    return x.map_elements(lambda series: butterworth_bandpass_np(series.to_numpy(), N, Wn, fs))

    
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
    
    def butterworth_hp(self, N, Wn, fs) -> pl.Expr:
        return butterworth_lowpass(self._expr)
    
    def butterworth_lp(self, N, Wn, fs) -> pl.Expr:
        return butterworth_highpass(self._expr)
    
    