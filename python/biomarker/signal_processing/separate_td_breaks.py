import numpy as np


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
