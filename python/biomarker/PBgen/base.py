# pyright: reportPrivateImportUsage=false
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
from sklearn.model_selection import StratifiedKFold, KFold

from model.base import get_dynamics_model
from dataset.struct_dataset import combine_struct_by_field


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


def group_pb_cross_asym(
    vec_output,
    features,
    idx_used: list,
    idx_break: npt.NDArray[np.int_],
    str_metric: str,
    max_width: int,
    width: int,
    top_k: int,
):

    # first find peaks in given metric
    vec_metric = combine_struct_by_field(vec_output, str_metric)
    assert ~np.any(np.isnan(vec_metric)), "NaNs found in metric"
    assert vec_metric.shape[0] == features.shape[1], "metric and features do not match"

    # find the peaks and also exclude the ones already used
    vec_idx_peak = signal.find_peaks(  # type: ignore
        vec_metric, height=np.percentile(vec_metric, 75), width=width / 2
    )[0]
    if len(idx_used) > 0:
        vec_idx_peak = np.setdiff1d(vec_idx_peak, np.concatenate(idx_used))
    assert len(vec_idx_peak) > 0, "No peaks in metric for single frequencies found"

    # now sort the peaks by metric performance
    vec_metric_peak = vec_metric[vec_idx_peak]
    idx_sort = np.argsort(vec_metric_peak)[::-1]
    vec_idx_peak = vec_idx_peak[idx_sort]

    # choose the top K peaks and proceed
    vec_idx_peak = vec_idx_peak[:top_k]

    # now also obtain all possible corresponding combinations around the peak
    vec_pb_full = []  # list of all candidate power bands
    vec_idx_peak_pb = []  # which power band corresponds to which peak
    for i in range(len(vec_idx_peak)):
        idx_peak = vec_idx_peak[i]
        vec_pb_curr = get_all_pb(idx_peak, max_width, idx_break, features.shape[1])

        # now append to outer list
        vec_idx_peak_pb.append(np.ones((len(vec_pb_curr),)) * i)
        vec_pb_full.extend(vec_pb_curr)
    vec_idx_peak_pb = np.concatenate(vec_idx_peak_pb, axis=0)

    return vec_pb_full, vec_idx_peak_pb


def correct_pb_feature_dim(
    features: npt.NDArray,
    idx_feature: list | int | npt.NDArray[np.int_],
    idx_used: list,
    n_iter: int,
):
    # TODO: verify validity of this function
    # organize the indexing for which power bin to extract from full feature matrix
    if isinstance(idx_feature, list):
        idx_feature = np.stack(idx_feature, axis=0)

    # obtain the next candidate power band
    if isinstance(idx_feature, int):
        features_sub = features[:, [idx_feature], ...]
        features_sub = (
            np.expand_dims(features_sub, axis=1)
            if len(features.shape) == 2
            else features_sub
        )

    # if multiple frequency bins in current poewr band
    else:
        # if only one power band selected
        if len(idx_feature) == 1:
            features_sub = features[:, idx_feature, ...]
            features_sub = (
                np.expand_dims(features_sub, axis=1)
                if len(features_sub.shape) == 2
                else features_sub
            )
        # if multiple power bands already exist
        else:
            features_sub = np.sum(features[:, idx_feature, ...], axis=1, keepdims=True)
            features_sub = (
                np.expand_dims(features_sub, axis=1)
                if len(features_sub.shape) == 2
                else features_sub
            )
    assert 2 <= len(features_sub.shape) <= 3, "Features should not be more than 3D"

    # now loop through existing power bands already selected
    if len(idx_used) > 0:
        features_used_sub = []
        for j in range(len(idx_used)):
            features_used_sub.append(
                np.sum(features[:, idx_used[j], ...], axis=1, keepdims=True)
            )
        features_used_sub = np.concatenate(features_used_sub, axis=1)

        if len(features_used_sub.shape) == 2:
            features_used_sub = np.expand_dims(features_used_sub, axis=-1)
        assert (
            2 <= len(features_used_sub.shape) <= 3
        ), "Features should not be more than 3D"

        features_sub = np.concatenate([features_used_sub, features_sub], axis=1)

    assert features_sub.shape[1] == n_iter
    assert 2 <= len(features_sub.shape) <= 3, "Features should not be more than 3D"

    return features_sub
