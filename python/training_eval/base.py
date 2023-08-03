# pyright: reportPrivateImportUsage=false
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold, KFold

from model.base import get_dynamics_model

_VEC_MODEL_DYNAMICS_ONLY = get_dynamics_model()


def correct_data_dim(str_model, features_sub: npt.NDArray,):
    # create empty list for holding output and sanity check
    assert features_sub.shape[0] > 0, "Input features should not be empty"

    # cases where model doesn't support dynamics
    if str_model not in _VEC_MODEL_DYNAMICS_ONLY:
        if len(features_sub.shape) > 2:
            features_sub_out = features_sub.copy().reshape(features_sub.shape[0], -1)
        else:
            features_sub_out = features_sub.copy()

    # cases where the model only takes in dynamics
    elif str_model in _VEC_MODEL_DYNAMICS_ONLY:
        if len(features_sub.shape) == 2:
            features_sub_out = np.expand_dims(
                features_sub, axis=-1
            )  # note axis=-1 here since data should be in shape (batch_size, n_channel, n_time)
        else:
            features_sub_out = features_sub.copy()
    else:
        raise ValueError("Model not found")

    return features_sub_out


def get_valid_data(
    features_train,
    y_class_train,
    features_test,
    y_class_test,
    bool_torch,
    n_fold,
    bool_use_strat_kfold,
    random_seed,
):
    # define the validation set splitter - note it might not get called
    # also note that every iteration the split is different
    if bool_use_strat_kfold:
        skf_valid = StratifiedKFold(
            n_splits=n_fold - 1, shuffle=True, random_state=random_seed
        )
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

    # perform sanity checks
    for features_curr in vec_features:
        assert np.all(
            features_curr.shape[1:] == vec_features[0].shape[1:]
        ), "Features should have the same shape"
    for y_class_curr in vec_y_class:
        assert np.all(
            y_class_curr.shape[1:] == vec_y_class[0].shape[1:]
        ), "Labels should have the same shape"

    return vec_features, vec_y_class
