import numpy as np


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
    
