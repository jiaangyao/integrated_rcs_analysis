# pylint: disable=no-member
import copy
import typing as tp

import ray
import tqdm
import numpy as np
import scipy.stats as stats
from easydict import EasyDict as edict

from biomarker.training.kfold_cv_training import kfold_cv_training
from biomarker.training.group_pb import group_pb_nested_cross_asym
from utils.combine_struct import combine_struct_by_field


def sfs_feature_sweep(features, y_class, y_stim, idx_feature, idx_used, n_class, n_fold, n_iter, str_model,
                      random_seed):
    # organize the features for the individual power bands
    features_sub = features[:, idx_feature, ...][:, None]
    if len(idx_used) > 0:
        features_used_sub = []
        for j in range(len(idx_used)):
            features_used_sub.append(np.sum(features[:, idx_used[j], ...], axis=1, keepdims=True))
        features_used_sub = np.concatenate(features_used_sub, axis=1)

        features_sub = np.concatenate([features_used_sub, features_sub], axis=1)
    assert features_sub.shape[1] == n_iter

    output = kfold_cv_training(features_sub, y_class, y_stim, n_class=n_class, n_fold=n_fold,
                               str_model=str_model, random_seed=random_seed)
    return output


@ ray.remote
def sfs_feature_sweep_ray(*args, **kwargs):
    """Ray wrapper for sfs_feature_sweep

    Args:
        args (tuple): arguments for sfs_feature_sweep
        kwargs (dict): keyword arguments for sfs_feature_sweep

    Returns:
        dict: output of sfs_feature_sweep
    """
    output = sfs_feature_sweep(*args, **kwargs)

    return output


def seq_forward_selection(features, y_class, y_stim, labels_cell, n_pass=5, n_fold=10, n_ch=3, width=5, 
                          max_width=10, top_k=10, top_best=5, str_model='LDA', str_metric='avg_auc', 
                          random_seed: tp.Optional[int]=None, 
                          bool_force_sfs_acc=False, 
                          bool_use_ray=True):

    # define initial variables
    idx_ch = (np.arange(n_ch)[:, None] + np.zeros((int(features.shape[1] / n_ch, )))[None, :]).reshape(-1).astype(
        dtype=np.int_)
    bool_used = np.zeros(features.shape[1], dtype=np.bool_)
    iter_used = np.zeros(features.shape[1])
    idx_all = np.arange(features.shape[1])
    idx_used = []

    # determine the number of classes
    n_class = len(np.unique(y_class)) * len(np.unique(y_stim))

    # create the full output structure
    output_init = dict()
    output_init['vec_pb_ord'] = []
    output_init['vec_acc'] = []
    output_init['vec_f1'] = []
    output_init['vec_conf_mat'] = []
    output_init['vec_auc'] = []
    output_fin = copy.deepcopy(output_init)
    orig_metric = None

    # start the main loop
    print('Sequential Forward Selection')
    for n_iter in tqdm.trange(1, n_pass + 1, leave=False, bar_format="{desc:<1.5}{percentage:3.0f}%|{bar:15}{r_bar}"):

        # loop through the features
        str_model_cv = 'LDA' if str_model == 'QDA' and n_iter == 1 else str_model
        if bool_use_ray:
            feature_handle = [sfs_feature_sweep_ray.remote(features, y_class, y_stim, i, idx_used, n_class, n_fold,
                                                           n_iter, str_model_cv, random_seed) for i in range(features.shape[1])]
            vec_output = ray.get(feature_handle)

            # vec_output = []
            # for i in range(features.shape[1]):
            #     vec_output.append(sfs_feature_sweep_ray(features, y_class, y_stim, i, idx_used, n_class, n_fold,
            #                                         n_iter, str_model_cv, random_seed))


        else:
            vec_output = []
            for i in tqdm.trange(features.shape[1], leave=False):
                vec_output.append(sfs_feature_sweep(features, y_class, y_stim, i, idx_used, n_class, n_fold,
                                                    n_iter, str_model_cv, random_seed))


        # obtain the initial AUC from first pass
        if len(output_fin['vec_pb_ord']) == 0:
            orig_metric = combine_struct_by_field(vec_output, str_metric)

        # find breaks in spectra based on past bands
        idx_avail = idx_all[~bool_used]
        idx_break = idx_avail[np.where(np.diff(idx_avail) > 1)[0]]

        # add breaks based on transition between channels
        idx_break = np.unique(np.concatenate([idx_break, np.where(np.diff(idx_ch) == 1)[0]]))
        str_model_sfs = 'LDA' if str_model == 'QDA' and n_iter == 1 else str_model
        str_metric_sfs = 'avg_acc' if bool_force_sfs_acc else str_metric
        vec_output_sfs = group_pb_nested_cross_asym(vec_output, features, idx_used, idx_avail, y_class, y_stim,
                                                    idx_break, n_class=n_class, n_fold=n_fold, width=width,
                                                    max_width=max_width, top_k=top_k, top_best=top_best,
                                                    str_model=str_model_sfs, str_metric=str_metric_sfs,
                                                    random_seed=random_seed)

        # find the best feature
        vec_metric_sfs = combine_struct_by_field(vec_output_sfs, str_metric)
        idx_sort = np.argsort(vec_metric_sfs)[::-1]
        vec_output_sfs = [vec_output_sfs[i] for i in idx_sort]

        if len(output_fin['vec_pb_ord']) == 0:
            output_init['vec_pb_ord'] = [idx_all[vec_output_sfs[i]['pb_best']] for i in range((len(vec_output_sfs)))]
            output_init['vec_acc'] = [vec_output_sfs[i]['avg_acc'] for i in range((len(vec_output_sfs)))]
            output_init['vec_f1'] = [vec_output_sfs[i]['avg_f1'] for i in range((len(vec_output_sfs)))]
            output_init['vec_conf_mat'] = [vec_output_sfs[i]['avg_conf_mat'] for i in range((len(vec_output_sfs)))]
            output_init['vec_auc'] = [vec_output_sfs[i]['avg_auc'] for i in range((len(vec_output_sfs)))]

        # combine the power bands
        output_fin['vec_pb_ord'].append(idx_all[vec_output_sfs[0]['pb_best']])
        output_fin['vec_acc'].append(vec_output_sfs[0]['avg_acc'])
        output_fin['vec_f1'].append(vec_output_sfs[0]['avg_f1'])
        output_fin['vec_conf_mat'].append(vec_output_sfs[0]['avg_conf_mat'])
        output_fin['vec_auc'].append(vec_output_sfs[0]['avg_auc'])

        # update the boolean mask for keeping track of used features
        iter_used[idx_all[vec_output_sfs[0]['pb_best']]] = n_iter
        bool_used[idx_all[vec_output_sfs[0]['pb_best']]] = True

        # update the list of used indices and test for non-overlap
        idx_used.append(idx_all[vec_output_sfs[0]['pb_best']])
        # assert(len(np.unique(np.concatenate(idx_used))) == len(np.concatenate(idx_used))), 'Overlap in power band!'

        # update the iteration counter
        n_iter += 1

    # organize the output
    pb_lim = np.stack([[output_fin['vec_pb_ord'][i][0], output_fin['vec_pb_ord'][i][-1]]
                       for i in range(len(output_fin['vec_pb_ord']))], axis=0)
    n_pb_lim = 5 if top_k > 5 else top_k
    pb_init_lim = np.stack([[output_init['vec_pb_ord'][i][0], output_init['vec_pb_ord'][i][-1]]
                            for i in range(n_pb_lim)], axis=0)

    # parse the power band labels
    vec_str_ch = [x[0] for x in labels_cell]
    vec_pb = [x[1] for x in labels_cell]
    pb_width = stats.mode(np.diff(vec_pb), keepdims=True)[0][0]

    # obtain the final power bands
    output_init['sinPB'] = []
    output_fin['sfsPB'] = []
    for i in range(len(output_fin['vec_pb_ord'])):
        # form the SFS PB and the single PB
        sinPB_curr = [vec_str_ch[pb_init_lim[i, 0]], vec_pb[pb_init_lim[i, 0]] - pb_width/2,
                      vec_pb[pb_init_lim[i, 1]] + pb_width/2]
        sfsPB_curr = [vec_str_ch[pb_lim[i, 0]], vec_pb[pb_lim[i, 0]] - pb_width/2,
                      vec_pb[pb_lim[i, 1]] + pb_width/2]
        assert (vec_str_ch[pb_lim[i, 0]] == vec_str_ch[pb_lim[i, 1]]), 'Channels do not match!'
        assert (vec_str_ch[pb_init_lim[i, 0]] == vec_str_ch[pb_init_lim[i, 1]]), 'Channels do not match!'

        output_init['sinPB'].append(sinPB_curr)
        output_fin['sfsPB'].append(sfsPB_curr)

    # return the outputs
    return output_fin, output_init, iter_used, orig_metric
