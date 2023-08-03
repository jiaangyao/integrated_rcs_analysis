"""
This script contains the HJ method developed by Lauren Hammer and Jiaang Yao for power band selection
"""

# pyright: reportPrivateImportUsage=false
import copy
import typing as tp
import itertools

import ray
import tqdm
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from omegaconf import DictConfig
import wandb

from model.pipeline import get_model_params
from .base import (
    correct_pb_feature_dim,
    group_pb_cross_asym,
)
from ._hj_pb_filtering import get_pb_filt_features, pb_filt_wrapper
from ._hj_pb_growing import get_pb_grow_features, pb_grow_wrapper
from dataset.struct_dataset import combine_struct_by_field
from utils.wandb_utils import wandb_logging_sfs_inner

# TODO: move all constants to constant directory
_MAX_NUMEBER_OF_FINAL_PB = 5


def hj_pb_method(
    features,
    y_class,
    y_stim,
    idx_ch,
    idx_all,
    idx_used,
    bool_used,
    n_iter,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    wandb_pb_filt: wandb.Table | None,
    wandb_pb_grow: wandb.Table | None,
    str_model="LDA",
    str_metric="avg_auc",
    n_candidate_peak=10,
    width=5,
    max_width=10,
    bool_force_acc=False,
    random_seed: int | None = None,
    n_fold=10,
    bool_use_strat_kfold=True,
    bool_use_ray=True,
    bool_use_gpu: bool = False,
    n_cpu_per_process: int | float = 1,
    n_gpu_per_process: float = 0.1,
    bool_use_batch: bool = False,
    batch_size: int = 1,
    bool_tune_hyperparams: bool = False,
    bool_use_wandb: bool = False,
    bool_use_lightweight_wandb: bool = False,
    bool_verbose: bool = True,
):
    """
    In the n-th iteration, perform the HJ method using current iteration of data
    """
    # perform quick sanity check
    assert np.allclose(
        idx_all[bool_used], idx_used
    ), "Variables tracking used indices are inconsistent"

    # first find all possible candidates for running algorithm
    # find breaks in spectra based on past bands
    idx_avail = idx_all[~bool_used]
    idx_break = idx_avail[np.where(np.diff(idx_avail) > 1)[0]]

    # now obtain the features for PB filtering step
    (
        vec_features_filt,
        str_model_pb,
        model_cfg_pb,
        trainer_cfg_pb,
    ) = get_pb_filt_features(
        features,
        idx_avail,
        idx_used,
        n_iter,
        model_cfg,
        trainer_cfg,
        str_model,
        bool_use_gpu,
        n_gpu_per_process,
        bool_tune_hyperparams,
    )

    # run PB filtering
    vec_output, wandb_pb_filt = pb_filt_wrapper(
        vec_features_filt,
        y_class,
        y_stim,
        model_cfg_pb,
        trainer_cfg_pb,
        n_fold=n_fold,
        str_model=str_model_pb,
        bool_use_strat_kfold=bool_use_strat_kfold,
        random_seed=random_seed,
        bool_use_ray=bool_use_ray,
        n_cpu_per_process=n_cpu_per_process,
        n_gpu_per_process=n_gpu_per_process,
        bool_use_batch=bool_use_batch,
        batch_size=batch_size,
        bool_use_wandb=bool_use_wandb,
        n_iter=n_iter,
        wandb_table=wandb_pb_filt,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
        bool_verbose=bool_verbose,
    )

    # obtain the initial AUC from first pass
    orig_metric = combine_struct_by_field(vec_output, str_metric) if n_iter == 1 else None

    # now obtain the features for PB growing step
    vec_features_pb_sub, vec_pb_full, vec_idx_peak_pb = get_pb_grow_features(
        vec_output,
        features,
        idx_ch,
        idx_used,
        idx_break,
        n_iter,
        str_metric,
        width,
        max_width,
        n_candidate_peak,
        bool_force_acc,
    )

    # next run PB growth
    vec_output_pb_grow, wandb_pb_grow = pb_grow_wrapper(
        vec_features_pb_sub,
        y_class,
        y_stim,
        model_cfg_pb,
        trainer_cfg_pb,
        n_fold=n_fold,
        str_model=str_model_pb,
        bool_use_strat_kfold=bool_use_strat_kfold,
        random_seed=random_seed,
        bool_use_ray=bool_use_ray,
        n_cpu_per_process=n_cpu_per_process,
        n_gpu_per_process=n_gpu_per_process,
        bool_use_batch=bool_use_batch,
        batch_size=batch_size,
        bool_use_wandb=bool_use_wandb,
        n_iter=n_iter,
        wandb_table=wandb_pb_grow,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
        bool_verbose=bool_verbose,
    )

    return vec_output_pb_grow, vec_pb_full, vec_idx_peak_pb, model_cfg_pb, trainer_cfg_pb, wandb_pb_filt, wandb_pb_grow, orig_metric
