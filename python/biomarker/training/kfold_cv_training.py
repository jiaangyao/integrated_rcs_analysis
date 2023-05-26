# pylint: disable=no-member
import typing as tp

import numpy as np
import numpy.typing as npt
import ray
from sklearn.model_selection import StratifiedKFold, KFold

from biomarker.training.get_model_params import get_model_params
from biomarker.training.correct_data_dim import correct_data_dim, get_valid_data
from biomarker.training.model_training import train_model, test_if_torch_model
from biomarker.training.correct_data_dim import correct_sfs_feature_dim
from utils.combine_labels import create_hashmap
from utils.combine_struct import combine_struct_by_field, create_output_struct, append_output_struct, arrayize_output_struct, comp_summary_output_struct

from utils.beam_search import beam_search


def kfold_cv_training(features_sub, y_class, y_stim, n_fold=10, str_model='LDA',
                      bool_use_strat_kfold=True, 
                      random_seed: int|None=0):
    """The inner function for obtaining the initial cross-validated metric 

    Args:
        features_sub (_type_): _description_
        y_class (_type_): _description_
        y_stim (_type_): _description_
        n_class (int, optional): _description_. Defaults to 4.
        n_fold (int, optional): _description_. Defaults to 10.
        str_model (str, optional): _description_. Defaults to 'LDA'.
        bool_use_ray (bool, optional): _description_. Defaults to True.
        bool_use_strat_kfold (bool, optional): _description_. Defaults to True.
        random_seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # create the training and test sets
    if bool_use_strat_kfold:
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
    else:
        skf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed)

    # create hashmap in advance from most general label
    hashmap = create_hashmap(y_class, y_stim)

    # check if validation data is needed
    bool_torch = test_if_torch_model(str_model)

    # obtain the number of classes
    n_class = len(np.unique(y_class)) * len(np.unique(y_stim))

    # create final output structure that will hold all information
    output = dict()

    # loop through the folds
    for train_idx, test_idx in skf.split(features_sub, y_class):
        # obtain data for train set
        features_train = features_sub[train_idx, ...]
        y_class_train = y_class[train_idx, ...]

        # obtain data for test set
        features_test = features_sub[test_idx, ...]
        y_class_test = y_class[test_idx, ...]
        y_stim_test = y_stim[test_idx, ...]

        # organize the various features for training
        vec_features, vec_y_class = get_valid_data(features_train, y_class_train, features_test, y_class_test,
                                                   bool_torch, n_fold, bool_use_strat_kfold, random_seed)

        # now define and train the model
        output_curr = train_model(vec_features, vec_y_class, y_stim_test,
                                  n_class=n_class, str_model=str_model, hashmap=hashmap)

        # append the variables to outer list
        if len(output.keys()) == 0:
            output = append_output_struct(create_output_struct(output, output_curr), output_curr)
        else:
            output = append_output_struct(output, output_curr)

    # convert the list to numpy array and compute the stats
    output = arrayize_output_struct(output)

    # create the output structure
    output = comp_summary_output_struct(output)

    return output


@ ray.remote
def kfold_cv_training_ray(*args, **kwargs):
    """Ray wrapper for kfold_cv_training

    Args:
        args (tuple): arguments for kfold_cv_training
        kwargs (dict): keyword arguments for kfold_cv_training

    Returns:
        dict: output of kfold_cv_training
    """

    output = kfold_cv_training(*args, **kwargs)

    return output

