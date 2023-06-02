import numpy as np
import numpy.typing as npt


def get_all_pb(
    freq: int,
    max_width: int,
    idx_break: npt.NDArray[np.int_],
    size_freq: int,
) -> list[npt.NDArray[np.int_]]:
    """Get all possible power bands given a frequency and a maximum width

    Args:
        freq (int): index of frequency bin in the feature vector (comb of ch and freq)
        max_width (int): width of the power band in Hz - since all center freq are 1Hz apart
        idx_break (list[int]): index of discontinuity in frequency axis of the feature vector
        size_freq (int): number of elements in the frequency axis of the feature vector

    Returns:
        list[npt.NDArray[np.int_]]: list of candidate power bands to test
    """

    assert max_width > 0, "Max width must be greater than 0"

    vec_pbs = []
    for i in np.arange(0, max_width, 1):
        for offset in np.arange(-i, 1):
            # determine the upper and higher cutoff for the frequencies
            idx_low_cutoff = idx_break[idx_break < freq]
            idx_low_cutoff = -1 if len(idx_low_cutoff) == 0 else np.max(idx_low_cutoff)

            idx_high_cutoff = idx_break[idx_break >= freq]
            idx_high_cutoff = (
                size_freq - 1 if len(idx_high_cutoff) == 0 else np.min(idx_high_cutoff)
            )

            # now obtain the trimmed limits for the possible power band
            idx_untrimmed = freq + offset + np.arange(0, i + 1)
            idx_trimmed = idx_untrimmed[
                (idx_untrimmed > idx_low_cutoff) & (idx_untrimmed <= idx_high_cutoff)
            ]

            vec_pbs.append(idx_trimmed)

    return vec_pbs
