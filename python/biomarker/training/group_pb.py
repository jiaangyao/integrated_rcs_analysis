import typing as tp

import tqdm
import ray
import numpy as np
import scipy.signal as signal

from biomarker.training.kfold_cv_training import kfold_cv_sfs_search # type: ignore
from utils.combine_struct import combine_struct_by_field


def group_pb_nested_cross_asym(vec_output, features, idx_used, idx_search, y_class, y_stim,
                               idx_break, n_fold=10, width=5, max_width=10, top_k=10, top_best=5,
                               str_model='LDA', str_metric='avg_auc', bool_use_ray: bool=True, 
                               random_seed: tp.Optional[int]=None):
    
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

    # if choose to use ray for parallelization
    if bool_use_ray:
        results_handle = [kfold_cv_sfs_search.remote(features, vec_idx_peak[i], idx_used, y_class, y_stim, idx_break,
                                                    n_class=n_class, n_fold=n_fold, top_best=top_best, max_width=max_width, # type: ignore
                                                    str_model=str_model, str_metric=str_metric, random_seed=random_seed) # type: ignore
                        for i in range(len(vec_idx_peak))]
        vec_output_sfs = ray.get(results_handle)

    else:
        vec_output_sfs = []
        for i in tqdm.trange(len(vec_idx_peak), leave=False, bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}"):
            vec_output_sfs.append(kfold_cv_sfs_search(features, vec_idx_peak[i], idx_used, y_class, y_stim, idx_break,
                                                      n_class=n_class, n_fold=n_fold, top_best=top_best, max_width=max_width,
                                                      str_model=str_model, str_metric=str_metric, random_seed=random_seed))

    return vec_output_sfs
