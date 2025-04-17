import numpy as np


def combine_hist(vec_pb):
    """
    Deprecated function for combining runs from multiple runs in a histogram manner
    """

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


def beam_search(
    vec_pb,
    vec_metric,
    min_len=2,
    beam_width=500,
):
    """
    Untested function for combining runs from multiple runs in a beam search style manner
    """
    # empty variable to hold completed
    completed = []

    # initialize beam, each beam item consists of common arrays
    beam = []
    for i in range(len(vec_pb[0])):
        beam.append((vec_pb[0][i], vec_metric[0][i], [i]))

    # now initialize the beam search
    for curr_idx in np.arange(1, len(vec_pb)):
        successor_states = vec_pb[curr_idx]
        successors_scores = vec_metric[curr_idx]

        # now form the new beams based on existing beams and successors
        new_beam = []
        for j in np.arange(len(beam)):
            for k in np.arange(len(successor_states)):
                new_state = np.intersect1d(beam[j][0], successor_states[k])
                new_score = beam[j][1] + successors_scores[k]
                new_idx = beam[j][2] + [k]

                if len(new_state) > 0 and len(new_state) > min_len:
                    new_beam.append((new_state, new_score, new_idx))

        # now sort the new beam by score
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

        # update existing beams
        if len(new_beam) <= beam_width:
            beam = new_beam
        else:
            beam = new_beam[:beam_width]

        # now break if empty
        if len(beam) == 0:
            raise ValueError("Beam search failed")

    t1 = 1
