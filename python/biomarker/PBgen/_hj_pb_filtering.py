import typing as tp

import ray
import tqdm
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import wandb

from model.pipeline import get_model_params
from training_eval.pipeline import (
    kfold_cv_training,
    kfold_cv_training_ray,
    kfold_cv_training_ray_batch,
)
from utils.wandb_utils import wandb_logging_sfs_inner
from .base import correct_pb_feature_dim


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
):
    """Set up power band filtering (SFS feature sweeping) not using parallelization
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
        list[dict]: list of output dictionaries of same structure, each representing the output at a particular
        frequency bin
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
        if (
            bool_use_wandb
            and not bool_use_lightweight_wandb
            and wandb_table is not None
        ):
            wandb_table.add_data(
                idx_feature + 1,
                output_curr["avg_acc"],
                output_curr["avg_f1"],
                output_curr["avg_auc"],
                n_iter,
            )

        # # if use lightweight wandb, then log the simple output
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
        str_sfs="PB_Filtering",
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
) -> tp.Tuple[list[dict], wandb.Table | None]:
    """Set up power band filtering (SFS feature sweeping) using Ray for parallelization

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
        list[dict]: list of output dictionaries of same structure, each representing the output at a particular
        frequency bin
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
    vec_output = [dict()] * len(vec_features_sub)
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
        assert (
            len(vec_output[i].keys()) > 0
        ), "Output structure can't contain empty dictionary"

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


def get_pb_filt_features(
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
):
    # loop through the features
    vec_features_sub = [
        correct_pb_feature_dim(features, idx_feature, idx_used, n_iter)
        for idx_feature in idx_avail
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

    return vec_features_sub, str_model_cv, model_cfg_cv, trainer_cfg_cv


def pb_filt_wrapper(*args, **kwargs):
    # run parallelized version of feature sweep
    if kwargs["bool_use_ray"]:
        vec_output, wandb_table = sfs_feature_sweep_ray(
            *args,
            n_fold=kwargs["n_fold"],
            str_model=kwargs["str_model"],
            bool_use_strat_kfold=kwargs["bool_use_strat_kfold"],
            random_seed=kwargs["random_seed"],
            n_cpu_per_process=kwargs["n_cpu_per_process"],
            n_gpu_per_process=kwargs["n_gpu_per_process"],
            bool_use_batch=kwargs["bool_use_batch"],
            batch_size=kwargs["batch_size"],
            bool_use_wandb=kwargs["bool_use_wandb"],
            n_iter=kwargs["n_iter"],
            wandb_table=kwargs["wandb_table"],
            bool_use_lightweight_wandb=kwargs["bool_use_lightweight_wandb"],
        )

    # run serial version of feature sweep
    else:
        vec_output, wandb_table = sfs_feature_sweep(
            *args,
            n_fold=kwargs["n_fold"],
            str_model=kwargs["str_model"],
            bool_use_strat_kfold=kwargs["bool_use_strat_kfold"],
            random_seed=kwargs["random_seed"],
            bool_use_wandb=kwargs["bool_use_wandb"],
            n_iter=kwargs["n_iter"],
            wandb_table=kwargs["wandb_table"],
            bool_use_lightweight_wandb=kwargs["bool_use_lightweight_wandb"],
            bool_verbose=kwargs["bool_verbose"],
        )

    return vec_output, wandb_table
