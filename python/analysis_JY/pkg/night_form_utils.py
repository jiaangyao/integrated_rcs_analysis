import pandas as pd
import numpy as np


def load_night_forms(p_night_forms, start_date, end_date):
    """
    Load the night forms within the specified time range (inclusive on both ends)

    :param p_night_forms:
    :param start_date:
    :param end_date:
    :return:
    """

    # load the nightly forms and convert time to pd.Timestamp
    df_night_forms = pd.read_excel(p_night_forms, sheet_name='RAW', header=[0, 1, 2])
    df_night_forms.loc[:, ('general patient info', 'day', 'day')] = df_night_forms.loc[:, ('general patient info', 'day', 'day')].dt.tz_localize('America/Los_Angeles')

    # convert the start and end dates to pd.Timestamp
    start_date_timestamp = pd.Timestamp(start_date, tz='America/Los_Angeles')
    end_date_timestamp = pd.Timestamp(end_date, tz='America/Los_Angeles')

    # obtain the relevant rows in the original table
    idx_valid_row = np.logical_and((df_night_forms.loc[:, ('general patient info', 'day', 'day')] >= start_date_timestamp).to_numpy(),
                                   (df_night_forms.loc[:, ('general patient info', 'day', 'day')] <= end_date_timestamp).to_numpy())
    df_night_forms = df_night_forms.loc[idx_valid_row, :].reset_index(drop=True)

    return df_night_forms


def interp_night_forms(df_night_forms, time_pkg, bool_valid):
    # first subselect the dates from the night forms
    date_night_forms = df_night_forms.loc[:, ('general patient info', 'day', 'day')]
    df_night_forms_interp = []
    for i in range(len(date_night_forms)):
        curr_date = date_night_forms.iloc[i].date()
        n_pkg_curr_day = np.sum(time_pkg.dt.date == curr_date)

        # now make copies of the night forms
        curr_table = pd.concat([df_night_forms.iloc[i, :]] * n_pkg_curr_day, axis=1).T.reset_index(drop=True)
        curr_table.loc[:, ('general patient info', 'day', 'day')] = time_pkg[time_pkg.dt.date == curr_date].reset_index(drop=True)
        df_night_forms_interp.append(curr_table)

    # make the final copy and assert same shape
    df_night_forms_interp = pd.concat(df_night_forms_interp, axis=0).reset_index(drop=True)
    assert df_night_forms_interp.shape[0] == len(time_pkg)
    assert df_night_forms_interp.shape[1] == df_night_forms.shape[1]

    # finally obtain the valid rows only
    df_night_forms_interp.loc[np.logical_not(bool_valid), ('My symptoms')] = np.nan

    return df_night_forms_interp

