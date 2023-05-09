from datetime import datetime, timedelta
import glob

import pandas as pd
import numpy as np

from utils.parse_datetime import parse_dt_w_tz


def load_pkg_data(p_pkg):
    df_pkg_side = pd.read_csv(p_pkg)

    # convert the datetime column to pd.Timestamp
    df_pkg_side['Date_Time'] = pd.to_datetime(df_pkg_side['Date_Time'], yearfirst=True, format='%Y-%m-%d %H:%M:%S')
    df_pkg_side['Date_Time'] = df_pkg_side['Date_Time'].dt.tz_localize('America/Los_Angeles')

    # also obtain a table that only has valid entries
    idx_valid = np.logical_and(np.logical_not(df_pkg_side['Off_Wrist'].to_numpy()),
                               np.logical_not(df_pkg_side[['DKS', 'BKS', 'TS']].isna().any(axis=1).to_numpy()))
    df_pkg_side_valid = df_pkg_side.copy()
    df_pkg_side_valid.loc[np.logical_not(idx_valid), ['DKS', 'BKS', 'TS']] = np.nan

    # also throw away time of possible sleep periods (11PM-8AM)
    idx_sleep = np.logical_or(df_pkg_side_valid['Date_Time'].dt.hour >= 23, df_pkg_side_valid['Date_Time'].dt.hour <= 8)
    df_pkg_side_valid.loc[idx_sleep, ['DKS', 'BKS', 'TS']] = np.nan

    # obtain the list of valid indices
    bool_valid = np.logical_and(idx_valid, np.logical_not(idx_sleep))

    return df_pkg_side, df_pkg_side_valid, bool_valid


def average_pkg_data(df_pkg, avg_min):
    """


    :param df_pkg:
    :param avg_min:
    :return:
    """

    # find out sampling rate
    sampling_period = np.unique(np.diff(df_pkg['Seconds'].array))
    sampling_period = sampling_period[sampling_period > 0]
    assert len(sampling_period) == 1

    # now get a sense for which rows to select
    T_min = np.int(np.ceil(sampling_period[0] / 60))
    n_period = np.int(avg_min / T_min)
    avg_min_int = n_period * T_min
    avg_sec_int = avg_min_int * 60

    # get the start and end time
    init_start_time = df_pkg['Date_Time'].iloc[0]
    init_end_time = df_pkg['Date_Time'].iloc[-1]
    n_period_full = np.int((init_end_time - init_start_time).total_seconds() / avg_sec_int)

    # now obtain the downsampled pandas table
    vec_start_time = []
    vec_end_time = []
    df_pkg_redact = []
    for i in range(n_period_full):
        curr_start_time = init_start_time + timedelta(seconds=i*avg_sec_int)
        curr_end_time = curr_start_time + timedelta(seconds=avg_sec_int)

        # append current time
        vec_start_time.append(curr_start_time)
        vec_end_time.append(curr_end_time)

        # figure out indices corresponding to current window
        idx_curr_win = np.logical_and(df_pkg['Date_Time'].array >= curr_start_time, df_pkg['Date_Time'].array < curr_end_time)

        # form the current row
        curr_table = df_pkg[idx_curr_win].copy()
        curr_row = curr_table.iloc[0].copy()

        # compute the values for all metrics
        idx_valid_curr_win = np.logical_and(np.logical_not(curr_table['Off_Wrist'].to_numpy()),
                                            np.logical_not(curr_table[['DKS', 'BKS', 'TS']].isna().any(axis=1).to_numpy()))

        if np.sum(idx_valid_curr_win) == 0:
            curr_row['DKS'] = np.nan
            curr_row['BKS'] = np.nan
            curr_row['TS'] = np.nan
        else:
            curr_row['DKS'] = curr_table.loc[idx_valid_curr_win, 'DKS'].mean()
            curr_row['BKS'] = curr_table.loc[idx_valid_curr_win, 'BKS'].mean()
            curr_row['TS'] = curr_table.loc[idx_valid_curr_win, 'TS'].mean()
            curr_row['Off_Wrist'] = False

        # next create the redacted table
        df_pkg_redact.append(curr_row)

    # convert redacted array into actual pandas dataframe
    df_pkg_redact = pd.concat(df_pkg_redact, axis=1).transpose().reset_index(drop=True)

    # also obtain the table with only valid entries
    idx_valid = np.logical_and(np.logical_not(df_pkg_redact['Off_Wrist'].to_numpy()),
                               np.logical_not(df_pkg_redact[['DKS', 'BKS', 'TS']].isna().any(axis=1).to_numpy()))
    df_pkg_redact_valid = df_pkg_redact.copy()
    df_pkg_redact_valid.loc[np.logical_not(idx_valid), ['DKS', 'BKS', 'TS']] = np.nan

    # also throw away time of possible sleep periods (11PM-8AM)
    idx_sleep = np.logical_or(df_pkg_redact_valid['Date_Time'].dt.hour >= 23, df_pkg_redact_valid['Date_Time'].dt.hour <= 8)
    df_pkg_redact_valid.loc[idx_sleep, ['DKS', 'BKS', 'TS']] = np.nan

    # obtain the list of valid indices
    bool_valid = np.logical_and(idx_valid, np.logical_not(idx_sleep))

    return df_pkg_redact, df_pkg_redact_valid, bool_valid
