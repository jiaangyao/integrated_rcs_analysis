# pylint: disable=no-member
import typing as tp

import numpy as np
import numpy.typing as npt
import ray
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
from easydict import EasyDict as edict

from biomarker.training.get_model_params import get_model_params
from biomarker.training.correct_data_dim import correct_data_dim, get_valid_data
from biomarker.training.model_infrastructure import get_model
from biomarker.training.torch_model_infrastructure import get_model_ray
from utils.combine_labels import combine_labels, create_hashmap
from utils.combine_struct import combine_struct_by_field
from utils.get_all_pb import get_all_pb
from utils.beam_search import beam_search
from utils.combine_hist import combine_hist


def kfold_cv_training(features_sub, y_class, y_stim, n_class=4, n_fold=10, str_model='LDA',
                      bool_use_ray=True, bool_use_strat_kfold=True, random_seed=0):
    # create the training and test sets
    if bool_use_strat_kfold:
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
    else:
        skf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed)

    # create hashmap in advance from most general label
    hashmap = create_hashmap(y_class, y_stim)

    # check if validation data is needed
    bool_torch = True if str_model in ['MLP', 'RNN'] else False

    # create list for holding the various variables
    vec_acc = []
    vec_conf_mat = []
    vec_auc = []
    vec_f1 = []

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
        acc, f1, conf_mat, roc, auc = train_model(features_train, features_test, y_class_train, y_class_test, y_stim_test,
                                                  str_model, hashmap, n_class, features_valid, y_class_valid)

        # append the variables to outer list
        vec_acc.append(acc)
        vec_conf_mat.append(conf_mat)
        vec_auc.append(auc)
        vec_f1.append(f1)

    # convert the list to numpy array and compute the stats
    vec_acc = np.stack(vec_acc, axis=0)
    vec_conf_mat = np.stack(vec_conf_mat, axis=0)
    vec_auc = np.stack(vec_auc, axis=0)
    vec_f1 = np.stack(vec_f1, axis=0)

    # create the output structure
    output = dict()
    output['avg_acc'] = np.mean(vec_acc)
    output['avg_conf_mat'] = np.sum(vec_conf_mat, axis=0) / np.sum(vec_conf_mat)
    output['avg_f1'] = float(np.mean(vec_f1))
    output['avg_auc'] = float(np.mean(vec_auc))
    output['std_f1'] = float(np.std(vec_f1))
    output['std_auc'] = float(np.std(vec_auc))

    return output


@ ray.remote # type: ignore
def kfold_cv_sfs_search_sin_pb(features, pb, idx_train, idx_test, idx_used, y_class_train, y_class_test, y_stim_test,
                               str_model, hashmap, n_class, n_fold, random_seed):
    # get the features for the current power band (also sum over the power axis)
    features_sub_train = np.sum(features[idx_train[:, None], pb, ...], axis=1)
    features_sub_test = np.sum(features[idx_test[:, None], pb, ...], axis=1)

    # if not the initial iteration of search, get the k folds for features used
    if len(idx_used) > 0:
        # now group the existing power bands
        features_used_train = []
        features_used_test = []

        for i in range(len(idx_used)):
            features_used_train.append(np.sum(features[idx_train[:, None], idx_used[i], ...], axis=1))
            features_used_test.append(np.sum(features[idx_test[:, None], idx_used[i], ...], axis=1))
        features_used_train = np.stack(features_used_train, axis=1)
        features_used_test = np.stack(features_used_test, axis=1)

        features_train = np.concatenate((features_used_train, features_sub_train[:, None]), axis=1)
        features_test = np.concatenate((features_used_test, features_sub_test[:, None]), axis=1)
    else:
        features_train = features_sub_train[:, None]
        features_test = features_sub_test[:, None]

    # include dynamcis if needed
    if len(features_train.shape) == 3:
        features_train = features_train.reshape(features_train.shape[0], -1)
        features_test = features_test.reshape(features_test.shape[0], -1)

    # check if validation data is needed
    bool_torch = True if str_model in ['MLP', 'RNN'] else False

    # now form the training and validation sets
    features_train, y_class_train, features_valid, y_class_valid = \
        get_valid_data(features_train, y_class_train, bool_torch, n_fold, random_seed)

    # now define and train the model
    acc, f1, conf_mat, roc, auc = train_model(features_train, features_test,
                                              y_class_train, y_class_test, y_stim_test,
                                              str_model, hashmap, n_class, features_valid, y_class_valid)

    # define output structure
    output = dict()
    output['acc'] = acc
    output['f1'] = f1
    output['conf_mat'] = conf_mat
    output['auc'] = auc
    output['roc'] = roc

    return output


@ray.remote
def kfold_cv_sfs_search(features, idx_peak, idx_used, y_class, y_stim, idx_break, max_width=10,
                        n_class=4, top_best=5, n_fold=10, str_model='LDA', str_metric='avg_auc',
                        random_seed=0):
    # set up for k-fold cross validation
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
    # skf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
    vec_pb_full = get_all_pb(idx_peak, max_width, idx_break, features.shape[1])

    # create output structure first
    output = dict()
    output['vec_acc'] = []
    output['vec_f1'] = []
    output['vec_conf_mat'] = []
    output['vec_auc'] = []
    output['vec_pb_best'] = []

    # create hashmap in advance from most general label
    hashmap = create_hashmap(y_class, y_stim)

    # start the training
    for idx_train, idx_test in skf.split(features, y_class):
        y_class_train = y_class[idx_train, ...]
        y_class_test = y_class[idx_test, ...]
        y_stim_test = y_stim[idx_test, ...]

        # now proceed with sequential forward selection
        pb_search_handle = [kfold_cv_sfs_search_sin_pb.remote(features, pb, idx_train, idx_test, idx_used,
                                                              y_class_train, y_class_test, y_stim_test,
                                                              str_model, hashmap, n_class, n_fold,
                                                              random_seed) for pb in vec_pb_full]
        vec_output_best_currk = ray.get(pb_search_handle)

        # now get the various metrics
        avg_acc = combine_struct_by_field(vec_output_best_currk, 'acc')
        avg_f1 = combine_struct_by_field(vec_output_best_currk, 'f1')
        avg_conf_mat = combine_struct_by_field(vec_output_best_currk, 'conf_mat')
        avg_auc = combine_struct_by_field(vec_output_best_currk, 'auc')

        # now obtain the best power band
        idx_max_metric = np.argsort(eval(str_metric))[::-1][:top_best]
        vec_pb_best_currk = [vec_pb_full[i] for i in idx_max_metric]
        output['vec_pb_best'].append(vec_pb_best_currk)
        output['vec_acc'].append(avg_acc[idx_max_metric])
        output['vec_f1'].append(avg_f1[idx_max_metric])
        output['vec_conf_mat'].append(avg_conf_mat[idx_max_metric, ...])
        output['vec_auc'].append(avg_auc[idx_max_metric])            

    # now organize the accuracy data
    _, output['pb_best'] = combine_hist(output['vec_pb_best'])
    output['avg_acc'] = np.mean(np.stack([output['vec_acc'][i][0] for i in range(len(output['vec_acc']))], axis=0))
    output['avg_f1'] = np.mean(np.stack([output['vec_f1'][i][0] for i in range(len(output['vec_f1']))], axis=0))
    output['avg_conf_mat'] = np.mean(
        np.stack([output['vec_conf_mat'][i][0, ...] for i in range(len(output['vec_conf_mat']))], axis=0), axis=0)
    output['avg_auc'] = np.mean(np.stack([output['vec_auc'][i][0] for i in range(len(output['vec_auc']))], axis=0))

    return output


def train_model(vec_features: list[npt.NDArray], vec_y_class: list[npt.NDArray], 
                y_stim_test: npt.NDArray, n_class: int=4, str_model: str='LDA', 
                hashmap: dict[str, int]|None=None), bool_use_ray) -> dict[str, npt.NDArray]:
    
    # obtain the dimensionality
    n_input = vec_features[0].shape[1]
    n_class_model = len(np.unique(vec_y_class[0]))

    # now define the model parameters and correct the feature dimensionality
    model_params, train_params, bool_torch = get_model_params(str_model, n_input, n_class_model)
    vec_features = correct_data_dim(str_model, vec_features)

    # obtain the parameters for training and append it to the configuraiton

    # now define the model
    if bool_torch:
        # unpack the features first
        features_train, features_valid, features_test = vec_features

        # now start the model training
        model = get_model_ray(str_model, model_params)

        # model.train.remote(features_train, y_class_train, valid_data=features_valid, valid_label=y_class_valid,
        #                    **train_params)
        model.train(features_train, y_class_train, valid_data=features_valid, valid_label=y_class_valid, **train_params)

        # next generate the predictions
        # y_class_pred = ray.get(model.predict.remote(features_test))
        y_class_pred = model.predict(features_test)
        if y_class_pred.ndim > 1:
            y_class_pred = np.argmax(y_class_pred, axis=1)
        # acc = ray.get(model.get_accuracy.remote(features_test, y_class_test))
        acc = model.get_accuracy(features_test, y_class_test)
        f1 = f1_score(y_class_test, y_class_pred)

    else:
        model = get_model(str_model, model_params)
        model.train(features_train, y_class_train)

        y_class_pred = model.predict(features_test)
        if y_class_pred.ndim > 1:
            y_class_pred = np.argmax(y_class_pred, axis=1)
        acc = model.get_accuracy(features_test, y_class_test)
        f1 = f1_score(y_class_test, y_class_pred)

    # create the big confusion matrix
    full_label_test, _ = combine_labels(y_class_test, y_stim_test, hashmap=hashmap)
    full_label_pred, _ = combine_labels(y_class_pred, y_stim_test, hashmap=hashmap)
    if hashmap is not None:
        conf_mat = confusion_matrix(full_label_test, full_label_pred, labels=list(hashmap.values()))
    else:
        conf_mat = confusion_matrix(full_label_test, full_label_pred)
    assert np.all(np.array(conf_mat.shape) == n_class), 'Confusion matrix is not the right size'

    # estimate the ROC curve
    if bool_torch:
        # y_class_pred_scores = ray.get(model.predict_proba.remote(features_test))
        y_class_pred_scores = model.predict_proba(features_test)
    else:
        y_class_pred_scores = model.predict_proba(features_test)
    roc = roc_curve(y_class_test, y_class_pred_scores)
    auc = roc_auc_score(y_class_test, y_class_pred_scores)

    return acc, f1, conf_mat, roc, auc
