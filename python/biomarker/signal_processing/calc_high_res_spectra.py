import scipy.signal as signal
import numpy as np

from biomarker.signal_processing.separate_td_breaks import separate_td_breaks
from biomarker.signal_processing.separate_by_stim import separete_by_stim
from biomarker.signal_processing.extract_freq_features import extract_freq_features


def calc_high_res_spectra(
    data_label,
    label_time_domain,
    fs,
    fft_len,
    interval,
    update_rate,
    stim_level,
    ch2use,
):
    # create the end variables to return
    vec_features_full = []
    vec_labels_class_full = []
    vec_labels_stim_full = []
    labels_cell_full = None
    vec_times_full = []

    # loop through the class labels
    for i in range(len(np.unique(label_time_domain))):
        print(".", end="")

        # if fft_len is not specified, use closest power of 2 for one second
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
