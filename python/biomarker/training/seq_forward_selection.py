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
import pandas as pd

from biomarker.training.model_initialize import get_model_params
from biomarker.training.correct_data_dim import (
    correct_sfs_feature_dim,
    group_pb_cross_asym,
)
from biomarker.training.kfold_cv_training import (
    kfold_cv_training,
    kfold_cv_training_ray,
    kfold_cv_training_ray_batch,
)
from utils.combine_struct import combine_struct_by_field
from utils.combine_hist import combine_hist
from utils.wandb_utils import wandb_logging_sfs_inner

_MAX_NUMEBER_OF_FINAL_PB = 5


def sfs_feature_sweep(
    vec_features_sub: list[npt.NDArray],
    y_class: npt.NDArray,
    y_stim: npt.NDArray,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    n_fold: int,
    str_model: str,
    bool_use_strat_kfold: bool,
    random_seed: int | None,
    bool_use_wandb: bool,
    n_iter: int,
    wandb_table: wandb.Table | None = None,
    bool_use_lightweight_wandb: bool = False,
    bool_verbose: bool = True,
) -> list[dict]:
    """Set up SFS feature sweeping not using parallelization
    Args:
        vec_features_sub (list[npt.NDArray]): list of organized features for training
        y_class (npt.NDArray): class label
        y_stim (npt.NDArray): stim level label
        model_cfg (DictConfig | dict): model configuration loaded via hydra
        trainer_cfg (DictConfig | dict): trainer configuration loaded via hydra
        n_fold (int): number of folds for cross validation
        str_model (str): name of the model to use, latest list in model_infrastructure and torch_model_infrastructure
        bool_use_strat_kfold (bool): whether to use stratified kfold as cross validation scheme
        random_seed (int | None): random seed for performing the cross validation, if None then no seed is used
        bool_use_wandb (bool): whether to use wandb for logging
        n_iter (int): index of current iteration
        wandb_table (wandb.Table | None, optional): wandb table to log the output. Defaults to None.
        bool_use_lightweight_wandb (bool, optional): whether to use lightweight wandb logging. Defaults to False.
        bool_verbose (bool, optional): whether to print out progress bar. Defaults to True.

    Returns:
        list[dict]: list of output dictionaries of same structure, each representing the output at a particular frequeney bin
    """

    vec_output = []
    for idx_feature in tqdm.trange(
        len(vec_features_sub),
        leave=False,
        desc="SFS1",
        bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        disable=not bool_verbose,
    ):
        # obtain output from current iteration (vec_metrics across CV folds)
        output_curr = kfold_cv_training(
            vec_features_sub[idx_feature],
            y_class,
            y_stim,
            idx_feature,
            model_cfg,
            trainer_cfg,
            n_fold=n_fold,
            str_model=str_model,
            bool_use_strat_kfold=bool_use_strat_kfold,
            random_seed=random_seed,
        )

        # append to outer list
        vec_output.append(output_curr)

        # incrementally update wandb table if already initialized
        if bool_use_wandb and not bool_use_lightweight_wandb and wandb_table is not None:
            wandb_table.add_data(
                idx_feature + 1,
                output_curr["avg_acc"],
                output_curr["avg_f1"],
                output_curr["avg_auc"],
                n_iter,
            )
            
        # # otherwise if use lightweight wandb, then log the simple output
        # elif bool_use_lightweight_wandb:
        #     log_dict = {
        #         "SFS1/center_freq": idx_feature + 1,
        #         "SFS1/acg_acc": output_curr["avg_acc"],
        #         "SFS1/acg_auc": output_curr["avg_auc"],
        #     }
        #     wandb.log(log_dict)

    # perform logging using wandb
    wandb_table = wandb_logging_sfs_inner(
        vec_output=vec_output,
        wandb_table=wandb_table,
        n_iter=n_iter,
        str_sfs="SFS1",
        bool_use_wandb=bool_use_wandb,
        bool_use_ray=False,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
    )

    return vec_output, wandb_table


def sfs_feature_sweep_ray(
    vec_features_sub: list[npt.NDArray],
    y_class: npt.NDArray,
    y_stim: npt.NDArray,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    n_fold: int,
    str_model: str,
    bool_use_strat_kfold: bool,
    random_seed: int | None,
    n_cpu_per_process: int | float,
    n_gpu_per_process: int | float,
    bool_use_batch: bool,
    batch_size: int,
    bool_use_wandb: bool,
    n_iter: int,
    wandb_table: wandb.Table | None = None,
    bool_use_lightweight_wandb: bool = False,
) -> list[dict]:
    """Set up SFS feature sweeping using Ray for parallelization

    Args:
        vec_features_sub (list[npt.NDArray]): list of organized features for training
        y_class (npt.NDArray): class label
        y_stim (npt.NDArray): stim level label
        model_cfg (DictConfig | dict): model configuration loaded via hydra
        trainer_cfg (DictConfig | dict): trainer configuration loaded via hydra
        n_fold (int): number of folds for cross validation
        str_model (str): name of the model to use, latest list in model_infrastructure and torch_model_infrastructure
        bool_use_strat_kfold (bool): whether to use stratified kfold as cross validation scheme
        random_seed (int | None): random seed for performing the cross validation, if None then no seed is used
        n_cpu_per_process (int | float): number of cpu per process
        n_gpu_per_process (int | float): number of gpu per process
        bool_use_batch (bool): whether to use batching for ray training
        batch_size (int): batch size for batching the features for ray training
        bool_use_wandb (bool): whether to use wandb for logging
        n_iter (int): index of current iteration
        wandb_table (wandb.Table | None, optional): wandb table to log the output. Defaults to None.
        bool_use_lightweight_wandb (bool, optional): whether to use lightweight wandb logging. Defaults to False.

    Returns:
        list[dict]: list of output dictionaries of same structure, each representing the output at a particular frequeney bin
    """

    # if use batching
    if bool_use_batch:
        feature_handle = []
        vec_idx_feature = np.arange(len(vec_features_sub))
        for i in range(0, len(vec_features_sub), batch_size):
            feature_handle.append(
                kfold_cv_training_ray_batch.options(
                    num_cpus=n_cpu_per_process,
                    num_gpus=n_gpu_per_process,
                ).remote(
                    vec_features_sub[i : i + batch_size],
                    y_class,
                    y_stim,
                    vec_idx_feature[i : i + batch_size],
                    model_cfg,
                    trainer_cfg,
                    n_fold=n_fold,
                    str_model=str_model,
                    bool_use_strat_kfold=bool_use_strat_kfold,
                    random_seed=random_seed,
                )
            )

    else:
        feature_handle = [
            kfold_cv_training_ray.options(
                num_cpus=n_cpu_per_process,
                num_gpus=n_gpu_per_process,
            ).remote(
                vec_features_sub[idx_feature],
                y_class,
                y_stim,
                idx_feature,
                model_cfg,
                trainer_cfg,
                n_fold=n_fold,
                str_model=str_model,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
            )
            for idx_feature in range(len(vec_features_sub))
        ]

    # incorporate ray.wait() especially for the result function
    vec_output = [None] * len(vec_features_sub)
    while len(feature_handle) > 0:
        done_id, feature_handle = ray.wait(feature_handle, num_returns=1)
        if bool_use_batch:
            vec_output_done = [*ray.get(done_id[0])]
            for output_done in vec_output_done:
                vec_output[output_done["idx_feature"]] = output_done

            # vec_output =  itertools.chain(*ray.get(feature_handle))
        else:
            output_done = ray.get(done_id[0])
            vec_output[output_done["idx_feature"]] = output_done

            # vec_output = ray.get(feature_handle)

    # final sanity check
    for i in range(len(vec_output)):
        assert vec_output[i] is not None

    # perform logging using wandb
    wandb_table = wandb_logging_sfs_inner(
        vec_output=vec_output,
        wandb_table=wandb_table,
        n_iter=n_iter,
        str_sfs="SFS1",
        bool_use_wandb=bool_use_wandb,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
        bool_use_ray=True,
    )

    return vec_output, wandb_table


def sfs_pb_sweep(
    vec_features_pb_sub: list[npt.NDArray],
    y_class: npt.NDArray,
    y_stim: npt.NDArray,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    n_fold: int,
    str_model: str,
    bool_use_strat_kfold: bool,
    random_seed: int | None,
    bool_use_wandb: bool,
    n_iter: int,
    wandb_table: wandb.Table | None = None,
    bool_use_lightweight_wandb: bool = False,
    bool_verbose: bool = True,
) -> list[dict]:
    # form the list of output
    vec_output_sfs = []
    for idx_feature in tqdm.trange(
        len(vec_features_pb_sub),
        leave=False,
        desc="SFS2",
        bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        disable=not bool_verbose,
    ):
        # obtain output from current iteration (vec_metrics across CV folds)
        output_curr = kfold_cv_training(
            vec_features_pb_sub[idx_feature],
            y_class,
            y_stim,
            idx_feature,
            model_cfg,
            trainer_cfg,
            n_fold=n_fold,
            str_model=str_model,
            bool_use_strat_kfold=bool_use_strat_kfold,
            random_seed=random_seed,
        )

        # append to the outer list
        vec_output_sfs.append(output_curr)

        # incrementally update wandb table if already initialized
        if bool_use_wandb and wandb_table is not None:
            wandb_table.add_data(
                idx_feature + 1,
                output_curr["avg_acc"],
                output_curr["avg_f1"],
                output_curr["avg_auc"],
                n_iter,
            )
            
        # # otherwise if use lightweight wandb, then log the simple output
        # elif bool_use_lightweight_wandb:
        #     log_dict = {
        #         "SFS2/freq_index": idx_feature + 1,
        #         "SFS2/acg_acc": output_curr["avg_acc"],
        #         "SFS2/acg_auc": output_curr["avg_auc"],
        #     }
        #     wandb.log(log_dict)

    # perform logging using wandb
    wandb_table = wandb_logging_sfs_inner(
        vec_output=vec_output_sfs,
        wandb_table=wandb_table,
        n_iter=n_iter,
        str_sfs="SFS2",
        bool_use_wandb=bool_use_wandb,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
        bool_use_ray=False,
    )

    return vec_output_sfs, wandb_table


def sfs_pb_sweep_ray(
    vec_features_pb_sub: list[npt.NDArray],
    y_class: npt.NDArray,
    y_stim: npt.NDArray,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    n_fold: int,
    str_model: str,
    bool_use_strat_kfold: bool,
    random_seed: int | None,
    n_cpu_per_process: int | float,
    n_gpu_per_process: int | float,
    bool_use_batch: bool,
    batch_size: int,
    bool_use_wandb: bool,
    n_iter: int,
    wandb_table: wandb.Table | None = None,
    bool_use_lightweight_wandb: bool = False,
) -> list[dict]:
    # if use batching
    if bool_use_batch:
        pb_handle = []
        vec_idx_feature = np.arange(len(vec_features_pb_sub))
        for i in range(0, len(vec_features_pb_sub), batch_size):
            pb_handle.append(
                kfold_cv_training_ray_batch.options(
                    num_cpus=n_cpu_per_process,
                    num_gpus=n_gpu_per_process,
                ).remote(
                    vec_features_pb_sub[i : i + batch_size],
                    y_class,
                    y_stim,
                    vec_idx_feature[i : i + batch_size],
                    model_cfg,
                    trainer_cfg,
                    n_fold=n_fold,
                    str_model=str_model,
                    bool_use_strat_kfold=bool_use_strat_kfold,
                    random_seed=random_seed,
                )
            )

    else:
        pb_handle = [
            kfold_cv_training_ray.options(
                num_cpus=n_cpu_per_process,
                num_gpus=n_gpu_per_process,
            ).remote(
                vec_features_pb_sub[idx_feature],
                y_class,
                y_stim,
                idx_feature,
                model_cfg,
                trainer_cfg,
                n_fold=n_fold,
                str_model=str_model,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
            )
            for idx_feature in range(len(vec_features_pb_sub))
        ]

    # incorporate ray.wait() especially for the result function
    vec_output_sfs = [None] * len(vec_features_pb_sub)
    while len(pb_handle) > 0:
        done_id, pb_handle = ray.wait(pb_handle, num_returns=1)
        if bool_use_batch:
            vec_output_done = [*ray.get(done_id[0])]
            for output_done in vec_output_done:
                vec_output_sfs[output_done["idx_feature"]] = output_done

            # vec_output_sfs = itertools.chain(*ray.get(pb_handle))

        else:
            output_done = ray.get(done_id[0])
            vec_output_sfs[output_done["idx_feature"]] = output_done

            # vec_output_sfs = ray.get(pb_handle)

    # final sanity check
    for i in range(len(vec_output_sfs)):
        assert vec_output_sfs[i] is not None

    # perform logging using wandb
    wandb_table = wandb_logging_sfs_inner(
        vec_output=vec_output_sfs,
        wandb_table=wandb_table,
        n_iter=n_iter,
        str_sfs="SFS2",
        bool_use_wandb=bool_use_wandb,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
        bool_use_ray=True,
        
    )

    return vec_output_sfs, wandb_table


def seq_forward_selection(
    features,
    y_class,
    y_stim,
    labels_cell,
    model_cfg: DictConfig | dict,
    trainer_cfg: DictConfig | dict,
    n_ch: int = 3,
    str_model="LDA",
    str_metric="avg_auc",
    n_fin_pb=5,
    n_candidate_peak=10,
    n_candidate_pb=5,
    width=5,
    max_width=10,
    bool_force_sfs_acc=False,
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
    """_summary_

    Args:
        features (_type_): _description_
        y_class (_type_): _description_
        y_stim (_type_): _description_
        labels_cell (_type_): _description_
        model_cfg (DictConfig | dict): _description_
        trainer_cfg (DictConfig | dict): _description_
        n_ch (int, optional): _description_. Defaults to 3.
        str_model (str, optional): _description_. Defaults to "LDA".
        str_metric (str, optional): _description_. Defaults to "avg_auc".
        n_fin_pb (int, optional): _description_. Defaults to 5.
        n_candidate_peak (int, optional): _description_. Defaults to 10.
        n_candidate_pb (int, optional): _description_. Defaults to 5.
        width (int, optional): _description_. Defaults to 5.
        max_width (int, optional): _description_. Defaults to 10.
        bool_force_sfs_acc (bool, optional): _description_. Defaults to False.
        random_seed (int | None, optional): _description_. Defaults to None.
        n_fold (int, optional): _description_. Defaults to 10.
        bool_use_strat_kfold (bool, optional): _description_. Defaults to True.
        bool_use_ray (bool, optional): _description_. Defaults to True.
        bool_use_gpu (bool, optional): _description_. Defaults to False.
        n_cpu_per_process (int | float, optional): _description_. Defaults to 1.
        n_gpu_per_process (float, optional): _description_. Defaults to 0.1.
        bool_use_batch (bool, optional): _description_. Defaults to False.
        batch_size (int, optional): _description_. Defaults to 1.
        bool_tune_hyperparams (bool, optional): _description_. Defaults to False.
        bool_use_wandb (bool, optional): _description_. Defaults to False.
        bool_verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    # define initial variables
    idx_ch = (
        (
            np.arange(n_ch)[:, None]
            + np.zeros(
                (
                    int(
                        features.shape[1] / n_ch,
                    )
                )
            )[None, :]
        )
        .reshape(-1)
        .astype(dtype=np.int_)
    )
    bool_used = np.zeros(features.shape[1], dtype=np.bool_)
    iter_used = np.zeros(features.shape[1])
    idx_all = np.arange(features.shape[1])
    idx_used = []

    # create the full output structure
    output_init = dict()
    output_init["vec_pb_ord"] = []
    output_init["vec_acc"] = []
    output_init["vec_f1"] = []
    output_init["vec_conf_mat"] = []
    output_init["vec_auc"] = []
    output_fin = copy.deepcopy(output_init)
    orig_metric = None

    # initialize table for SFS1 and SFS2 if using wandb
    wandb_table_sfs1 = None
    wandb_table_sfs2 = None

    # start the main loop
    for n_iter in tqdm.trange(
        1,
        n_fin_pb + 1,
        leave=False,
        desc="ITER",
        bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        disable=not bool_verbose,
    ):
        # loop through the features
        # first form the features vector
        vec_features_sub = [
            correct_sfs_feature_dim(features, idx_feature, idx_used, n_iter)
            for idx_feature in range(features.shape[1])
        ]

        # optionally override the model and trainer configs
        str_model_cv = "LDA" if str_model == "QDA" and n_iter == 1 else str_model
        if str_model_cv != str_model:
            model_cfg_cv, trainer_cfg_cv = get_model_params(
                str_model_cv,
                bool_use_gpu=bool_use_gpu,
                n_gpu_per_process=n_gpu_per_process,
                bool_tune_hyperparams=bool_tune_hyperparams,
            )
        else:
            model_cfg_cv = model_cfg
            trainer_cfg_cv = trainer_cfg

        # run parallelized version of feature sweep
        if bool_use_ray:
            vec_output, wandb_table_sfs1 = sfs_feature_sweep_ray(
                vec_features_sub,
                y_class,
                y_stim,
                model_cfg_cv,
                trainer_cfg_cv,
                n_fold=n_fold,
                str_model=str_model_cv,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
                n_cpu_per_process=n_cpu_per_process,
                n_gpu_per_process=n_gpu_per_process,
                bool_use_batch=bool_use_batch,
                batch_size=batch_size,
                bool_use_wandb=bool_use_wandb,
                n_iter=n_iter,
                wandb_table=wandb_table_sfs1,
                bool_use_lightweight_wandb=bool_use_lightweight_wandb,
            )

        # run serial version of feature sweep
        else:
            vec_output, wandb_table_sfs1 = sfs_feature_sweep(
                vec_features_sub,
                y_class,
                y_stim,
                model_cfg_cv,
                trainer_cfg_cv,
                n_fold=n_fold,
                str_model=str_model_cv,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
                bool_use_wandb=bool_use_wandb,
                n_iter=n_iter,
                wandb_table=wandb_table_sfs1,
                bool_use_lightweight_wandb=bool_use_lightweight_wandb,
                bool_verbose=bool_verbose,
            )

        # obtain the initial AUC from first pass
        if len(output_fin["vec_pb_ord"]) == 0:
            orig_metric = combine_struct_by_field(vec_output, str_metric)

        # find breaks in spectra based on past bands
        idx_avail = idx_all[~bool_used]
        idx_break = idx_avail[np.where(np.diff(idx_avail) > 1)[0]]

        # add breaks based on transition between channels
        idx_break = np.unique(
            np.concatenate([idx_break, np.where(np.diff(idx_ch) == 1)[0]])
        )
        str_model_sfs = "LDA" if str_model == "QDA" and n_iter == 1 else str_model
        str_metric_sfs = "avg_acc" if bool_force_sfs_acc else str_metric

        # now obtain all possible power bands to run analysis and form features
        vec_pb_full, vec_idx_peak_pb = group_pb_cross_asym(
            vec_output,
            features,
            y_class,
            y_stim,
            idx_used,
            idx_break,
            str_metric=str_metric_sfs,
            max_width=max_width,
            width=width,
            top_k=n_candidate_peak,
        )
        vec_features_pb_sub = [
            correct_sfs_feature_dim(features, pb, idx_used, n_iter)
            for pb in vec_pb_full
        ]

        # run parallelized version of feature combination sweep
        if bool_use_ray:
            vec_output_sfs_sweep, wandb_table_sfs2 = sfs_pb_sweep_ray(
                vec_features_pb_sub,
                y_class,
                y_stim,
                model_cfg_cv,
                trainer_cfg_cv,
                n_fold=n_fold,
                str_model=str_model_sfs,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
                n_cpu_per_process=n_cpu_per_process,
                n_gpu_per_process=n_gpu_per_process,
                bool_use_batch=bool_use_batch,
                batch_size=batch_size,
                bool_use_wandb=bool_use_wandb,
                n_iter=n_iter,
                wandb_table=wandb_table_sfs2,
                bool_use_lightweight_wandb=bool_use_lightweight_wandb,
            )

        # run serial version of feature combinationsweep
        else:
            vec_output_sfs_sweep, wandb_table_sfs2 = sfs_pb_sweep(
                vec_features_pb_sub,
                y_class,
                y_stim,
                model_cfg_cv,
                trainer_cfg_cv,
                n_fold=n_fold,
                str_model=str_model_sfs,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
                bool_use_wandb=bool_use_wandb,
                n_iter=n_iter,
                wandb_table=wandb_table_sfs2,
                bool_use_lightweight_wandb=bool_use_lightweight_wandb,
                bool_verbose=bool_verbose,
            )

        # TODO: turn this into a separate function
        # now obtain the top power bands (defined by n_fin_pb)
        vec_output_sfs = []
        for i in range(len(np.unique(vec_idx_peak_pb))):
            # loop through the peaks and for each peak create output structure
            idx_pb_curr = np.where(vec_idx_peak_pb == i)[0]
            vec_output_sfs_pb_curr = [vec_output_sfs_sweep[j] for j in idx_pb_curr]
            vec_pb_curr = [vec_pb_full[j] for j in idx_pb_curr]

            # now obtain the metric and sort
            vec_metric_pb_curr = combine_struct_by_field(
                vec_output_sfs_pb_curr, str_metric
            )
            idx_max_metric = np.argsort(vec_metric_pb_curr)[::-1][:n_candidate_pb]

            # now form the best power band from current peak
            # _, output_sfs['pb_best'] = combine_hist(vec_metric_pb_curr)
            output_sfs = vec_output_sfs_pb_curr[idx_max_metric[0]]
            output_sfs["pb_best"] = vec_pb_curr[idx_max_metric[0]]

            vec_output_sfs.append(output_sfs)

        # organize the different power bands by metric
        vec_metric_sfs = combine_struct_by_field(vec_output_sfs, str_metric)
        idx_sort = np.argsort(vec_metric_sfs)[::-1][:n_fin_pb]
        vec_output_sfs = [vec_output_sfs[i] for i in idx_sort]

        # initialize the initial output structure for the single power bands
        if len(output_fin["vec_pb_ord"]) == 0:
            output_init["vec_pb_ord"] = [
                idx_all[vec_output_sfs[i]["pb_best"]] for i in range(n_fin_pb)
            ]
            output_init["vec_acc"] = [
                vec_output_sfs[i]["avg_acc"] for i in range(n_fin_pb)
            ]
            output_init["vec_f1"] = [
                vec_output_sfs[i]["avg_f1"] for i in range(n_fin_pb)
            ]
            output_init["vec_conf_mat"] = [
                vec_output_sfs[i]["avg_conf_mat"] for i in range(n_fin_pb)
            ]
            output_init["vec_auc"] = [
                vec_output_sfs[i]["avg_auc"] for i in range(n_fin_pb)
            ]

        # for the SFS output, pick the top power band and proceed
        output_fin["vec_pb_ord"].append(idx_all[vec_output_sfs[0]["pb_best"]])
        output_fin["vec_acc"].append(vec_output_sfs[0]["avg_acc"])
        output_fin["vec_f1"].append(vec_output_sfs[0]["avg_f1"])
        output_fin["vec_conf_mat"].append(vec_output_sfs[0]["avg_conf_mat"])
        output_fin["vec_auc"].append(vec_output_sfs[0]["avg_auc"])

        # update the boolean mask for keeping track of used features
        iter_used[idx_all[vec_output_sfs[0]["pb_best"]]] = n_iter
        bool_used[idx_all[vec_output_sfs[0]["pb_best"]]] = True

        # update the list of used indices and test for non-overlap
        idx_used.append(idx_all[vec_output_sfs[0]["pb_best"]])
        # assert(len(np.unique(np.concatenate(idx_used))) == len(np.concatenate(idx_used))), 'Overlap in power band!'

        # update the iteration counter
        n_iter += 1

    # organize the output
    # first obtain the power band edges
    pb_lim = np.stack(
        [
            [output_fin["vec_pb_ord"][i][0], output_fin["vec_pb_ord"][i][-1]]
            for i in range(len(output_fin["vec_pb_ord"]))
        ],
        axis=0,
    )
    pb_init_lim = np.stack(
        [
            [output_init["vec_pb_ord"][i][0], output_init["vec_pb_ord"][i][-1]]
            for i in range(len(output_init["vec_pb_ord"]))
        ],
        axis=0,
    )
    # parse the power band labels
    vec_str_ch = [x[0] for x in labels_cell]
    vec_pb = [x[1] for x in labels_cell]
    pb_width = stats.mode(np.diff(vec_pb), keepdims=True)[0][0]  # type: ignore

    # obtain the final power bands
    # form the SFS power bands first
    output_fin["sfsPB"] = []
    for i in range(min([_MAX_NUMEBER_OF_FINAL_PB, len(output_fin["vec_pb_ord"])])):
        # form the PB
        sfsPB_curr = [
            vec_str_ch[pb_lim[i, 0]],
            vec_pb[pb_lim[i, 0]] - pb_width / 2,
            vec_pb[pb_lim[i, 1]] + pb_width / 2,
        ]
        assert (
            vec_str_ch[pb_lim[i, 0]] == vec_str_ch[pb_lim[i, 1]]
        ), "Channels do not match!"
        output_fin["sfsPB"].append(sfsPB_curr)

    # now form the single PB
    output_init["sinPB"] = []
    for i in range(min([_MAX_NUMEBER_OF_FINAL_PB, len(output_init["vec_pb_ord"])])):
        # form the PB
        sinPB_curr = [
            vec_str_ch[pb_init_lim[i, 0]],
            vec_pb[pb_init_lim[i, 0]] - pb_width / 2,
            vec_pb[pb_init_lim[i, 1]] + pb_width / 2,
        ]
        assert (
            vec_str_ch[pb_init_lim[i, 0]] == vec_str_ch[pb_init_lim[i, 1]]
        ), "Channels do not match!"
        output_init["sinPB"].append(sinPB_curr)

    # append np.nan in case any search didn't find all power bands
    # proceed with SFS output first
    n_fin_pb = (
        _MAX_NUMEBER_OF_FINAL_PB if n_fin_pb > _MAX_NUMEBER_OF_FINAL_PB else n_fin_pb
    )  # force most to be _MAX_NUMEBER_OF_FINAL_PB
    conf_mat = output_fin["vec_conf_mat"][0]
    if len(output_fin["vec_pb_ord"]) < n_fin_pb:
        n_missing = n_fin_pb - len(output_fin["vec_pb_ord"])
        output_fin["vec_pb_ord"] += [np.nan, np.nan] * n_missing
        output_fin["vec_acc"] += [np.nan] * n_missing
        output_fin["vec_f1"] += [np.nan] * n_missing
        output_fin["vec_conf_mat"] += [np.ones_like(conf_mat) * np.nan] * n_missing
        output_fin["vec_auc"] += [np.nan] * n_missing
        output_fin["sfsPB"] += [np.nan] * n_missing

    # now proceed with single PB output
    if len(output_init["vec_pb_ord"]) < n_fin_pb:
        n_missing = n_fin_pb - len(output_init["vec_pb_ord"])
        output_init["vec_pb_ord"] += [np.nan, np.nan] * n_missing
        output_init["vec_acc"] += [np.nan] * n_missing
        output_init["vec_f1"] += [np.nan] * n_missing
        output_init["vec_conf_mat"] += [np.ones_like(conf_mat) * np.nan] * n_missing
        output_init["vec_auc"] += [np.nan] * n_missing
        output_init["sfsPB"] += [np.nan] * n_missing

    # return the outputs
    return output_fin, output_init, iter_used, orig_metric, model_cfg_cv, trainer_cfg_cv
