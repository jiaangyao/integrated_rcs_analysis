import numpy as np


def combine_hist(vec_pb):
    # first obtain the range of testing
    vec_pb_flat = np.concatenate(
        [np.concatenate(vec_pb[i]) for i in range(len(vec_pb))]
    )
    min_pb = np.min(vec_pb_flat)
    max_pb = np.max(vec_pb_flat)

    # form the histograms
    hist_indiv = []
    bin_indiv = []
    for i in range(len(vec_pb)):
        hist, bin = np.histogram(
            np.concatenate(vec_pb[i]), bins=np.arange(min_pb, max_pb + 1)
        )

        hist_indiv.append(hist)
        bin_indiv.append(bin)

    assert (bin_indiv == bin_indiv[0]).all(), "Bins are not the same"
    hist = np.sum(np.stack(hist_indiv, axis=0), axis=0)

    # obtain the zero indices and find the max of non-zero counts
    m_not_zero = []
    zero_idx = np.where(hist == 0)[0]
    zero_idx = np.concatenate([np.array([-1]), zero_idx, np.array([len(hist)])])
    for i in range(len(zero_idx) - 1):
        temp = hist[(zero_idx[i] + 1) : zero_idx[i + 1]]
        if len(temp) == 0:
            m_not_zero.append(0)
        else:
            m_not_zero.append(np.max(temp))

    # then obtain the valid indices from the max segment
    idx_max = np.argmax(m_not_zero)
    hist = hist[(zero_idx[idx_max] + 1) : zero_idx[idx_max + 1]]
    edge = bin_indiv[0][(zero_idx[idx_max] + 1) : zero_idx[idx_max + 1]]

    # finally obtain where the histogram is greater than half of maximum counts
    bool_out = hist > (np.max(hist) / 2)
    hist = hist[bool_out]
    edge = edge[bool_out]

    return hist, edge
