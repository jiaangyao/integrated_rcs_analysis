import numpy as np

# TODO: put all constants into constants directory
_MAX_NUMBER_OF_FINAL_PB = 5


def get_idx_ch(n_ch, n_bins):
    """
    Get index of channel for all FFT bins
    """
    idx_ch = (
        (np.arange(n_ch)[:, None] + np.zeros((int(n_bins / n_ch)))[None, :])
        .reshape(-1)
        .astype(dtype=np.int_)
    )

    return idx_ch
