import numpy as np


def beam_search(vec_pb, vec_metric, min_len=2, beam_width=500):
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
            raise ValueError('Beam search failed')

    t1 = 1
