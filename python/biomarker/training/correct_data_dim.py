import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


_VEC_MODEL_DYNAMICS_ONLY = ['RNN']

def correct_data_dim(str_model, vec_features):
    # create empty list for holding output and sanity check
    assert(len(vec_features) > 0), 'Input features should not be empty'
    vec_features_out = []
        
    # cases where model doesn't support dynamics
    if str_model not in _VEC_MODEL_DYNAMICS_ONLY:
        if len(vec_features[0].shape) > 2:
            for features_curr in vec_features:
                assert np.all(features_curr.shape == vec_features[0].shape), 'Features should have the same shape'
                features_curr_reshape = features_curr.copy().reshape(features_curr.shape[0], -1)
                vec_features_out.append(features_curr_reshape)
    
    # cases where the model only takes in dynamics
    elif str_model in _VEC_MODEL_DYNAMICS_ONLY:
        if len(vec_features[0].shape) == 2:
            for features_curr in vec_features:
                assert np.all(features_curr.shape == vec_features[0].shape), 'Features should have the same shape'
                features_curr_reshape = np.expand_dims(features_curr, axis=1)
                vec_features_out.append(features_curr_reshape)
    else:
        raise ValueError('Model not found')
    
    return vec_features_out


def correct_sfs_sweep_feature_dim(features, idx_feature, idx_used, n_iter):

    # organize the features for the individual power bands
    features_sub = features[:, idx_feature, ...][:, None]
    if len(idx_used) > 0:
        features_used_sub = []
        for j in range(len(idx_used)):
            features_used_sub.append(np.sum(features[:, idx_used[j], ...], axis=1, keepdims=True))
        features_used_sub = np.concatenate(features_used_sub, axis=1)

        features_sub = np.concatenate([features_used_sub, features_sub], axis=1)
    assert features_sub.shape[1] == n_iter

    return features_sub


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