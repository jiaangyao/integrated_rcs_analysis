import numpy as np
import numpy.typing as npt
import scipy.signal as signal
from sklearn.model_selection import StratifiedKFold, KFold

from biomarker.training.torch_model_infrastructure import get_dynamics_model
from utils.combine_struct import combine_struct_by_field
from utils.get_all_pb import get_all_pb


_VEC_MODEL_DYNAMICS_ONLY = get_dynamics_model()


def correct_data_dim(str_model, vec_features):
    # create empty list for holding output and sanity check
    assert(len(vec_features) > 0), 'Input features should not be empty'
    vec_features_out = []
        
    # cases where model doesn't support dynamics
    if str_model not in _VEC_MODEL_DYNAMICS_ONLY:
        if len(vec_features[0].shape) > 2:
            for features_curr in vec_features:
                assert np.all(features_curr.shape[1:] == vec_features[0].shape[1:]), 'Features should have the same shape'
                features_curr_reshape = features_curr.copy().reshape(features_curr.shape[0], -1)
                vec_features_out.append(features_curr_reshape)
        else:
            vec_features_out = [features for features in vec_features]
    
    # cases where the model only takes in dynamics
    elif str_model in _VEC_MODEL_DYNAMICS_ONLY:
        if len(vec_features[0].shape) == 2:
            for features_curr in vec_features:
                assert np.all(features_curr.shape[1:] == vec_features[0].shape[1:]), 'Features should have the same shape'
                features_curr_reshape = np.expand_dims(features_curr, axis=1)
                vec_features_out.append(features_curr_reshape)
        else:
            vec_features_out = [features for features in vec_features]
    else:
        raise ValueError('Model not found')
    
    assert(len(vec_features_out) > 0), 'Output features should not be empty'
    
    return vec_features_out


def correct_sfs_feature_dim(features: npt.NDArray, 
                                  idx_feature: list|int|npt.NDArray[np.int_], 
                                  idx_used: list, 
                                  n_iter: int):

    # organize the indexing for which power bin to extract from full feature matrix
    if isinstance(idx_feature, list):
        idx_feature = np.stack(idx_feature, axis=0)

    # obtain the next candidate power band
    if isinstance(idx_feature, int):
        features_sub = features[:, [idx_feature], ...][:, None]
    else:
        if len(idx_feature) == 1:
            features_sub = features[:, idx_feature, ...][:, None]
        else:
            features_sub = np.sum(features[:, idx_feature, ...], axis=1, keepdims=True)[:, None]

        assert 2 <= len(features_sub.shape) <= 3, 'Features should not be more than 3D'
    
    # now loop through existing power bands already selected
    if len(idx_used) > 0:
        features_used_sub = []
        for j in range(len(idx_used)):
            features_used_sub.append(np.sum(features[:, idx_used[j], ...], axis=1, keepdims=True))
        features_used_sub = np.concatenate(features_used_sub, axis=1)

        if len(features_used_sub.shape) == 2:
            features_used_sub = np.expand_dims(features_used_sub, axis=-1)

        features_sub = np.concatenate([features_used_sub, features_sub], axis=1)
    assert features_sub.shape[1] == n_iter

    return features_sub


def group_pb_cross_asym(vec_output, features, y_class, y_stim, 
                        idx_used: list, idx_break: npt.NDArray[np.int_],
                        str_metric: str, max_width: int, width: int, top_k: int):
    # obtain the number of classes
    n_class = len(np.unique(y_class)) * len(np.unique(y_stim))

    # first find peaks in given metric
    vec_metric = combine_struct_by_field(vec_output, str_metric)
    assert ~np.any(np.isnan(vec_metric)), 'NaNs found in metric'
    assert vec_metric.shape[0] == features.shape[1], 'metric and features do not match'

    # find the peaks and also exclude the ones already used
    vec_idx_peak = signal.find_peaks(vec_metric, height=np.percentile(vec_metric, 75), width=width/2)[0]
    if len(idx_used) > 0:
        vec_idx_peak = np.setdiff1d(vec_idx_peak, np.concatenate(idx_used))
    assert len(vec_idx_peak) > 0, 'No peaks in metric for single frequencies found'

    # now sort the peaks by metric performance
    vec_metric_peak = vec_metric[vec_idx_peak]
    idx_sort = np.argsort(vec_metric_peak)[::-1]
    vec_idx_peak = vec_idx_peak[idx_sort]

    # choose the top K peaks and proceed
    vec_idx_peak = vec_idx_peak[:top_k]

    # now also obtain all possible correspondin
    vec_pb_full = []    # list of all candidate power bands
    vec_idx_peak_pb = []    # which power band corresponds to which peak
    for i in range(len(vec_idx_peak)):
        idx_peak = vec_idx_peak[i]
        vec_pb_curr = get_all_pb(idx_peak, max_width, idx_break, features.shape[1])

        # now append to outer list
        vec_idx_peak_pb.append(np.ones((len(vec_pb_curr), )) * i)
        vec_pb_full.extend(vec_pb_curr)
    vec_idx_peak_pb = np.concatenate(vec_idx_peak_pb, axis=0)

    return vec_pb_full, vec_idx_peak_pb


def get_valid_data(features_train, y_class_train, features_test, y_class_test, 
                   bool_torch, n_fold, bool_use_strat_kfold, random_seed):
    
    # define the validation set splitter - note it might not get called
    # also note that every iteration the split is different
    if bool_use_strat_kfold:
        skf_valid = StratifiedKFold(n_splits=n_fold - 1, shuffle=True, random_state=random_seed)
    else:
        skf_valid = KFold(n_splits=n_fold - 1, shuffle=True, random_state=random_seed)

    # now optionally split the data into validation set in addition to train and test
    if bool_torch:
        # for neural net then generate additional validation set for early stopping
        train_idx, valid_idx = next(skf_valid.split(features_train, y_class_train))

        # obtain data for train set and valid set
        features_valid = features_train[valid_idx, ...]
        y_class_valid = y_class_train[valid_idx, ...]

        # obtain data for valid set
        features_train = features_train[train_idx, ...]
        y_class_train = y_class_train[train_idx, ...]

        # now obtain the vector containing all features
        vec_features = [features_train, features_valid, features_test]
        vec_y_class = [y_class_train, y_class_valid, y_class_test]

    else:
        # for non-neural net then just use the train and test set
        vec_features = [features_train, features_test]
        vec_y_class = [y_class_train, y_class_test]

    return vec_features, vec_y_class