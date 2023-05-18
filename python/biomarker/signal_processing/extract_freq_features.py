from itertools import product

import numpy as np
import scipy.signal as signal


def extract_freq_features(sig, fs, T, chLab, NFFT, interval):
    # first trim the signal
    vec_features = []
    lab_cells_out = None
    # vec_labN = []

    for i in range(len(sig)):
        # trim the signal to have integer number of T*fs
        sig_curr = sig[i]
        len_trim = int(np.floor(sig_curr.shape[0] / (T * fs)) * (T * fs))
        sig_trim = sig_curr[0:len_trim, :]

        # next reshape before pwelch
        sig_reshape = sig_trim.reshape(-1, int(T * fs), sig_trim.shape[1]).transpose(1, 0, 2)
        assert np.allclose(sig_reshape[:, 0, 1], sig_trim[0 * int(T * fs):(0 + 1) * int(T * fs), 1])

        # perform pwelch
        f, pxx = signal.welch(sig_reshape, fs, window=signal.windows.hamming(int(NFFT)), # type: ignore
                              noverlap=int(np.floor(fs*(1-interval))), axis=0, nfft=int(NFFT),
                              detrend=False) # type: ignore

        features = np.transpose(pxx, (2, 0, 1)).reshape(-1, pxx.shape[1])
        assert np.allclose(pxx[:, 0, 2], features[2 * f.shape[0]:(2 + 1) * f.shape[0], 0])

        # create the labels for the features
        lab_cells = list(product(chLab, f))
        # lab_array = np.arange(len(f)).T[:, None] + (np.arange(len(chLab)) * len(f))[None, :]
        # lab_N = lab_array.reshape(-1)

        # append to outer array
        vec_features.append(features)

        # also get the labels
        if i == 0:
            lab_cells_out = lab_cells
        else:
            assert lab_cells_out is not None, 'Should be initialized'
            lab_cells_out = None if not all(x == y for x, y in zip(lab_cells, lab_cells_out)) else lab_cells_out

    # concatenate the features (freq * epoch)
    features = np.concatenate(vec_features, axis=1)
    assert lab_cells_out is not None

    return features, lab_cells_out
