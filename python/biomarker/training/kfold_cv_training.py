# pyright: reportPrivateImportUsage=false
import typing as tp

import numpy as np
import numpy.typing as npt
import ray
from sklearn.model_selection import StratifiedKFold, KFold
from omegaconf import DictConfig

from biomarker.training.model_initialize import initialize_model
from biomarker.training.correct_data_dim import correct_data_dim, get_valid_data
from biomarker.training.model_training import train_model
from utils.combine_labels import create_hashmap
from utils.combine_struct import (
    combine_struct_by_field,
    create_output_struct,
    append_output_struct,
    arrayize_output_struct,
    comp_summary_output_struct,
)

from utils.beam_search import beam_search


def kfold_cv_training(
    features_sub: npt.NDArray,
    y_class,
    y_stim,
    idx_feature,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    n_fold=10,
    str_model="LDA",
    bool_use_strat_kfold=True,
    random_seed: int | None = 0,
):
    # create the training and test sets
    if bool_use_strat_kfold:
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
    else:
        skf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed)

    # create hashmap in advance from most general label
    hashmap = create_hashmap(y_class, y_stim)

    # correct feature dimensionality and obtain the dimensionality for model parameter
    features_sub = correct_data_dim(str_model, features_sub)
    n_input = features_sub.shape[1]
    n_class_model = len(np.unique(y_class))

    # initialize the model for training
    model = initialize_model(
        str_model,
        model_cfg,
        n_input,
        n_class_model,
    )

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
        vec_features, vec_y_class = get_valid_data(
            features_train,
            y_class_train,
            features_test,
            y_class_test,
            model.bool_torch,
            n_fold,
            bool_use_strat_kfold,
            random_seed,
        )

        # now define and train the model
        output_curr = train_model(
            model,
            vec_features,
            vec_y_class,
            y_stim_test,
            trainer_cfg,
            n_class=n_class,
            hashmap=hashmap,
        )

        # append the variables to outer list
        if len(output.keys()) == 0:
            output = append_output_struct(
                create_output_struct(output, output_curr), output_curr
            )
        else:
            output = append_output_struct(output, output_curr)

    # convert the list to numpy array and compute the stats
    output = arrayize_output_struct(output)

    # create the output structure
    output = comp_summary_output_struct(output)

    # also append the index of current input
    output["idx_feature"] = idx_feature

    return output


def kfold_Cv_training_batch(
    vec_features_batch,
    y_class,
    y_stim,
    vec_idx_feature,
    model_cfg,
    trainer_cfg,
    **kwargs,
):
    vec_output = [
        kfold_cv_training(
            vec_features_batch[i],
            y_class,
            y_stim,
            vec_idx_feature[i],
            model_cfg,
            trainer_cfg,
            **kwargs,
        )
        for i in range(len(vec_features_batch))
    ]

    return vec_output


@ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
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


@ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
def kfold_cv_training_ray_batch(*args, **kwargs):
    """Ray wrapper for kfold_cv_training

    Args:
        args (tuple): arguments for kfold_cv_training
        kwargs (dict): keyword arguments for kfold_cv_training

    Returns:
        dict: output of kfold_cv_training
    """

    vec_output = kfold_Cv_training_batch(*args, **kwargs)

    return vec_output
