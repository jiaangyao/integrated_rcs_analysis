from itertools import product

import numpy as np
import scipy.signal as signal


def extract_rcs_feature_pwelch(
    data_label,
    label_time_domain,
    fs,
    fft_len,
    interval,
    update_rate,
    stim_level,
    ch2use,
):
    """Used to be calc_high_res_spectra in Lauren's code

    Args:
        data_label (_type_): _description_
        label_time_domain (_type_): _description_
        fs (_type_): _description_
        fft_len (_type_): _description_
        interval (_type_): _description_
        update_rate (_type_): _description_
        stim_level (_type_): _description_
        ch2use (_type_): _description_

    Returns:
        _type_: _description_
    """

    # initial print statement
    print("\nCalculating High Resolution Spectra", end="")

    # now calculate the high resolution spectra
    # sanity check first for stim level
    assert len(stim_level) > 0, "Stim level must be specified"

    # create the end variables to return
    vec_features_full = []
    vec_labels_class_full = []
    vec_labels_stim_full = []
    labels_cell_full = None
    vec_times_full = []

    # loop through the class labels
    for i in range(len(np.unique(label_time_domain))):
        print(".", end="")

        # if fft_len is not specified, use the closest power of 2 for one second
        if fft_len is None:
            fft_len = 1
        NFFT = fs * fft_len
        T = NFFT / fs + interval * update_rate

        # now obtain the data from all cortical channels
        keys = np.arange(data_label.shape[1])[
            data_label.columns.str.contains("key", case=False)
        ]
        keys = keys[ch2use]
        str_ch = [ch.split("_")[1] for ch in data_label.columns[keys]]

        # now identify all inconsistencies in the data
        assert data_label.shape[0] == label_time_domain.shape[0]
        data_label_curr = data_label[label_time_domain == i]
        vec_data_chunk_label = separate_td_breaks(data_label_curr, fs)
        vec_sig_chunk_label = [
            data_chunk_curr.iloc[:, keys].values
            for data_chunk_curr in vec_data_chunk_label
        ]
        vec_time_chunk_label = [
            data_chunk_curr.loc[:, "time"].values
            for data_chunk_curr in vec_data_chunk_label
        ]
        idx_stim = np.arange(data_label.shape[1])[
            data_label.columns.str.contains("stim", case=False)
        ]

        # now calculate the spectra
        vec_stim_chunk_label = [
            data_chunk_curr.iloc[:, idx_stim].values
            for data_chunk_curr in vec_data_chunk_label
        ]
        vec_sig_chunk_label_full = []
        vec_stim_chunk_label_full = []
        vec_time_chunk_label_full = []
        for j in range(len(vec_sig_chunk_label)):
            x, y, z = separete_by_stim(
                vec_sig_chunk_label[j], vec_stim_chunk_label[j], vec_time_chunk_label[j]
            )
            vec_sig_chunk_label_full.extend(x)
            vec_stim_chunk_label_full.extend(y)
            vec_time_chunk_label_full.extend(z)

        # now throw away chunks that are too short
        bool_too_short = [
            True if x.shape[0] < NFFT * T else False for x in vec_sig_chunk_label_full
        ]
        vec_sig_chunk_label_full = [
            x for x, y in zip(vec_sig_chunk_label_full, bool_too_short) if not y  # type: ignore
        ]
        vec_time_chunk_label_full = [
            x for x, y in zip(vec_time_chunk_label_full, bool_too_short) if not y  # type: ignore
        ]
        vec_stim_chunk_label_full = [
            x for x, y in zip(vec_stim_chunk_label_full, bool_too_short) if not y  # type: ignore
        ]

        # now loop through the stim level
        vec_features_stim = []
        vec_labels_stim = []
        vec_times_stim = []
        lab_cell_stim = None
        for j in range(len(stim_level)):
            bool_stim_match = [
                True if np.all(x == stim_level[j]) else False
                for x in vec_stim_chunk_label_full
            ]
            sigs_curr = [
                x for x, y in zip(vec_sig_chunk_label_full, bool_stim_match) if y  # type: ignore
            ]
            times_curr = [
                x for x, y in zip(vec_time_chunk_label_full, bool_stim_match) if y  # type: ignore
            ]

            # now try to extract the frequency features
            # TODO: veritfy the validity of this calculation
            features_curr, lab_cell_curr = extract_freq_features(
                sigs_curr, fs, T, str_ch, NFFT, interval
            )

            # now append the features and labels
            vec_features_stim.append(features_curr)
            if j == 0:
                lab_cell_stim = lab_cell_curr
            else:
                assert lab_cell_stim is not None, "Should be initialized"
                lab_cell_stim = (
                    None
                    if not all(x == y for x, y in zip(lab_cell_stim, lab_cell_curr))  # type: ignore
                    else lab_cell_curr
                )

            # quick sanity check
            assert lab_cell_stim is not None, "Difference in stim settings"

            # now append the class labels for classification and the times
            vec_times_stim.append(times_curr)
            vec_labels_stim.append(
                (np.ones(features_curr.shape[1]) * j).astype(np.int_)
            )

        # concatenate into a single array
        vec_features_stim = np.concatenate(vec_features_stim, axis=1)
        vec_labels_stim = np.concatenate(vec_labels_stim, axis=0)
        vec_labels_class_curr = (np.ones(vec_features_stim.shape[1]) * i).astype(
            np.int_
        )

        # now append everything to outer loop
        vec_features_full.append(vec_features_stim)
        vec_labels_class_full.append(vec_labels_class_curr)
        vec_labels_stim_full.append(vec_labels_stim)
        if i == 0:
            labels_cell_full = lab_cell_stim
        else:
            assert labels_cell_full is not None, "Should be initialized"
            assert lab_cell_stim is not None, "Should be initialized"
            labels_cell_full = (
                None
                if not all(x == y for x, y in zip(labels_cell_full, lab_cell_stim))  # type: ignore
                else lab_cell_stim
            )
        vec_times_full.append(vec_times_stim)

    # concatenate the output into a single array
    vec_features_full = np.concatenate(vec_features_full, axis=1)
    vec_labels_class_full = np.concatenate(vec_labels_class_full, axis=0)
    vec_labels_stim_full = np.concatenate(vec_labels_stim_full, axis=0)
    print("")

    # sanity check
    assert labels_cell_full is not None, "Difference in stim settings"

    return (
        vec_features_full,
        vec_labels_class_full,
        vec_labels_stim_full,
        labels_cell_full,
        vec_times_full,
    )


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
        sig_reshape = sig_trim.reshape(-1, int(T * fs), sig_trim.shape[1]).transpose(
            1, 0, 2
        )
        assert np.allclose(
            sig_reshape[:, 0, 1], sig_trim[0 * int(T * fs) : (0 + 1) * int(T * fs), 1]
        )

        # perform pwelch
        f, pxx = signal.welch(
            sig_reshape,
            fs,
            window=signal.windows.hamming(int(NFFT)),  # type: ignore
            noverlap=int(np.floor(fs * (1 - interval))),
            axis=0,
            nfft=int(NFFT),
            detrend=False,  # type: ignore
        )

        features = np.transpose(pxx, (2, 0, 1)).reshape(-1, pxx.shape[1])
        assert np.allclose(
            pxx[:, 0, 2], features[2 * f.shape[0] : (2 + 1) * f.shape[0], 0]
        )

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
            assert lab_cells_out is not None, "Should be initialized"
            lab_cells_out = (
                None
                if not all(x == y for x, y in zip(lab_cells, lab_cells_out))  # type: ignore
                else lab_cells_out
            )

    # concatenate the features (freq * epoch)
    features = np.concatenate(vec_features, axis=1)
    assert lab_cells_out is not None

    return features, lab_cells_out


def separate_td_breaks(data, fs):
    # reset index
    data.reset_index(drop=True, inplace=True)

    # identify all potential inconsistencies in the data
    time_diff = data["time"].diff()[2:].dt.microseconds / 1e6
    time_sample_diff = np.ones_like(time_diff) * (1 / fs)

    # obtain the indices of the non-continuous data
    idx_non_continuous = (
        np.where(np.logical_not(np.isclose(time_diff, time_sample_diff)))[0] + 1
    )
    idx_start = np.append(0, idx_non_continuous + 1)
    idx_end = np.append(idx_non_continuous, data.shape[0] - 1)
    assert len(idx_start) == len(idx_end)

    # now get all the chunks of data
    vec_data_chunk = []
    for i in range(len(idx_start)):
        # obtain start and end time and perform sanity check
        time_start = data.loc[idx_start[i], "time"]
        time_end = data.loc[idx_end[i], "time"]
        assert time_start < time_end
        if i != len(idx_start) - 1:
            assert data.loc[idx_end[i], "time"] < data.loc[idx_start[i + 1], "time"]

        # now split the data into chunks
        data_chunk_curr = data.iloc[idx_start[i] : (idx_end[i] + 1), :]
        vec_data_chunk.append(data_chunk_curr)

    return vec_data_chunk


def separete_by_stim(data, stim, time):
    # identify all potential inconsistencies with stim in the data
    stim_diff = np.diff(stim, axis=0)
    idx_non_continuous = np.where(np.logical_not(np.isclose(stim_diff, 0)))[0] + 1
    idx_start = np.append(0, idx_non_continuous + 1)
    idx_end = np.append(idx_non_continuous, data.shape[0] - 1)
    assert len(idx_start) == len(idx_end)

    if len(idx_non_continuous) == 0:
        return [data], [stim], [time]

    # now get all the chunks of data
    # TODO: fix this later
    return [data], [stim], [time]
    vec_data_chunk = []
    vec_stim_chunk = []
    vec_time_chunk = []
