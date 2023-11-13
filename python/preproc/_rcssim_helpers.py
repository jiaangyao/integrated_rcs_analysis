import numpy as np
import numpy.typing as npt

import rcs_sim as rcs


def rcssim_fft_wrapper(
    td_data: npt.NDArray,
    time_stamps: npt.NDArray,
    settings,
    gain,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in polars lists of times (DerivedTimes) and timedomain data (td_data), and returns an estimate of the embedded RCS fft vector.
    params:
    times: (np.ndarray of floats: unix timestamps) 1xn vector of DerivedTimes
    td_data: (np.ndarray of floats: millivolts) 1xn vector of time domain data
    settings: (dict) settings for device

    returns: data_fft_out: (np.ndarray of floats) Estimate of embedded FFT vector(s) (size: mxn) corresponding to td_data
            t_pb: (np.ndarray: unix timestamp) DerivedTime unix timestamp of FFT vector
    """
    hann_win = rcs.create_hann_window(
        settings["fft_size"], percent=settings["window_percent"]
    )

    td_rcs = rcs.transform_mv_to_rcs(td_data, gain)

    data_fft, t_pb = rcs.td_to_fft(
        td_rcs,
        time_stamps,
        settings["samplingRate"],
        settings["fft_size"],
        settings["fft_interval"],
        hann_win,
        interp_drops=False,
        output_in_mv=False,
        shift_timestamps_up_by_one_ind=True,
    )

    data_fft_out = rcs.fft_to_pb(
        data_fft,
        settings["samplingRate"],
        settings["fft_size"],
        settings["fft_bandFormationConfig"],
        input_is_mv=False,
    )

    return data_fft_out, t_pb
