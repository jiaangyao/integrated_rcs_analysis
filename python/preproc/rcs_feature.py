import numpy as np
import numpy.typing as npt
import polars as pl

from ._pwelch_psd_feature import extract_rcs_feature_pwelch
from ._rcssim_helpers import rcssim_fft_wrapper
from .time_domain_base import identity

def extract_td_data_df(df: pl.DataFrame, label_type: str):
    # epoching operation

    # TODO: make the output from this function into a numpy array
    # obtain the sampling rate from data
    vec_fs = df["samplerate"].unique()
    vec_fs_valid = vec_fs[np.logical_not(np.isnan(vec_fs))]
    assert len(vec_fs_valid) == 1
    fs = vec_fs_valid[0]

    # now obtain the data from all cortical channels
    vec_ch_neural = [True if "key" in ch.lower() else False for ch in df.columns]
    vec_str_ch = df.columns[vec_ch_neural]
    str_ch = [ch.split("_")[1] for ch in vec_str_ch]

    # obtain the boolean for all variables needed for analysis
    bool_col2use = np.logical_or.reduce(
        (
            df.columns.str.contains("time", case=False),
            df.columns.str.contains("key", case=False),
            df.columns.str.contains("stim", case=False),
        )
    )

    # obtain the number of cortical channels
    # TODO: make into a separate function - and understand what it's really doing
    new_boolK = df.columns[bool_col2use].str.contains("key", case=False)
    indK = np.where(new_boolK)[0]
    # now hardcode the column to choose from
    if len(indK) >= 3:
        ch2use = [0, 2, 3]
    else:
        ch2use = [0, 1]

    # organizing data
    if label_type.lower() == "med":
        data_td = df.loc[:, bool_col2use]
        label_td = df["MedState"].values
    elif label_type.lower() == "sleep":
        data_td = df.loc[:, bool_col2use]
        label_td = df["SleepStageBinary"].values
    else:
        raise NotImplementedError

    return data_td, label_td, fs, str_ch, ch2use


# TODO: check out the polarify decorator
def extract_rcs_feature(
    data_td: pl.DataFrame,
    # label_td,
    # fs,
    # ch2use,
    preproc_settings: dict,
    stim_level: list = [],
    fft_len=None,
    interval=0.05,
    update_rate=30,
    low_lim=2,
    high_lim=100,
    bool_use_dyna=False,
    n_dynamics=1,
    amp_gain=None,
    str_method="pwelch",
):
    
    # time domain functions
    data_td = identity(data_td)

    # extract features for training using welch's method
    if str_method == "pwelch":
        (
            vec_features_full,
            y_class,
            y_stim,
            labels_cell_full,
            vec_times_full,
        ) = extract_rcs_feature_pwelch(
            data_td,
            label_td,
            fs,
            fft_len,
            interval,
            update_rate,
            stim_level,
            ch2use,
        )
    else:
        raise NotImplementedError("rcssim feature extraction method not implemented")

    # now organize the features
    features, y_class, y_stim, labels_cell = organize_feature(
        vec_features_full,
        y_class,
        y_stim,
        labels_cell_full,
        low_lim,
        high_lim,
        bool_use_dyna,
        n_dynamics,
    )

    return features, y_class, y_stim, labels_cell, vec_times_full


def organize_feature(
    vec_features_full,
    y_class,
    y_stim,
    labels_cell_full,
    low_lim,
    high_lim,
    bool_use_dyna,
    n_dynamics,
):
    # print statement
    print("Auto Power Band Selection Initiation")

    # extract features corresponding to frequency range between low_lim and high_lim
    bool_lim = [
        True if (low_lim <= f[1] <= high_lim) else False for f in labels_cell_full
    ]
    features = vec_features_full[bool_lim, :].T

    # if using dynamics then reshape the features
    if bool_use_dyna:
        vec_features = []
        vec_y_class = []
        vec_y_stim = []
        for i in range(int(np.floor(features.shape[0] / n_dynamics))):
            features_curr = features[i * n_dynamics : (i + 1) * n_dynamics, :].T
            y_class_curr = y_class[i * n_dynamics : (i + 1) * n_dynamics]
            y_stim_curr = y_stim[i * n_dynamics : (i + 1) * n_dynamics]

            if np.all(y_class_curr == y_class_curr[0]) and np.all(
                y_stim_curr == y_stim_curr[0]
            ):
                vec_features.append(features_curr)
                vec_y_class.append(y_class_curr[0])
                vec_y_stim.append(y_stim_curr[0])

        # now make into a single list
        features = np.stack(vec_features, axis=0)
        y_class = np.stack(vec_y_class, axis=0)
        y_stim = np.stack(vec_y_stim, axis=0)

        assert features.shape[0] == y_class.shape[0] == y_stim.shape[0]

    # now reorganize the string for labels
    labels_cell = [x for x, y in zip(labels_cell_full, bool_lim) if y]  # type: ignore

    return features, y_class, y_stim, labels_cell
