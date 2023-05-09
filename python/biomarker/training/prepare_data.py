import scipy
import numpy as np

from biomarker.signal_processing.calc_high_res_spectra import calc_high_res_spectra


def prepare_data(data, stim_level, str_side='L', label_type='med', interval=0.05, update_rate=30,
                 low_lim=2, high_lim=100):
    print('Calculating High Resolution Spectra', end='')
    # first hardcode the side of the data and side of the opposite side
    str_contra_side = 'R' if str_side.lower() == 'l' else 'L'

    # also set up the parameters
    vec_fs = data['samplerate'].unique()
    vec_fs_valid = vec_fs[np.logical_not(np.isnan(vec_fs))]
    assert len(vec_fs_valid) == 1

    fs = vec_fs_valid[0]
    n_fft = fs

    # now obtain the data from all cortical channels
    vec_ch_cort = [True if 'key' in ch.lower() else False for ch in data.columns]
    vec_str_ch = data.columns[vec_ch_cort]
    str_ch = [ch.split('_')[1] for ch in vec_str_ch]

    # obtain the boolean for all variables needed for analysis
    bool_col2use = np.logical_or.reduce((data.columns.str.contains('time', case=False),
                                         data.columns.str.contains('key', case=False),
                                         data.columns.str.contains('stim', case=False)))

    new_boolK = data.columns[bool_col2use].str.contains('key', case=False)
    indK = np.where(new_boolK)[0]

    # now hardcode the column to choose from
    if len(indK) >= 3:
        ch2use = [0, 2, 3]
    else:
        ch2use = [0, 1]

    # organizing data
    if label_type.lower() == 'med':
        data_label = data.loc[:, bool_col2use]
        label_time_domain = data['MedState'].values
    else:
        raise NotImplementedError

    # now calculate the high resolution spectra
    stim_level_side = getattr(stim_level, str_side)
    vec_features_full, y_class, y_stim, labels_cell_full, vec_times_full = \
        calc_high_res_spectra(data_label, label_time_domain, fs, interval, update_rate, stim_level_side, ch2use)

    # now get features in the valid frequency range
    print('Auto Power Band Selection Initiation')
    bool_lim = [True if (low_lim <= f[1] <= high_lim) else False for f in labels_cell_full]
    features = vec_features_full[bool_lim, :].T
    labels_cell = [x for x, y in zip(labels_cell_full, bool_lim) if y]

    return features, y_class, y_stim, labels_cell, vec_times_full
