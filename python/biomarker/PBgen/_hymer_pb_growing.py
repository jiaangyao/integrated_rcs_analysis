# pyright: reportPrivateImportUsage=false
import typing as tp

import ray
import tqdm
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import wandb

from training_eval.pipeline import (
    kfold_cv_training,
    kfold_cv_training_ray,
    kfold_cv_training_ray_batch,
)
from utils.wandb_utils import wandb_logging_sfs_inner
from .base import correct_pb_feature_dim, group_pb_cross_asym


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
) -> tp.Tuple[list[dict], wandb.Table | None]:
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
        str_sfs="PB_Growing",
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
) -> tp.Tuple[list[dict], wandb.Table | None]:
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
    vec_output_sfs = [dict()] * len(vec_features_pb_sub)
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
        assert len(vec_output_sfs[i].keys()) > 0, "Output structure"

    # perform logging using wandb
    wandb_table = wandb_logging_sfs_inner(
        vec_output=vec_output_sfs,
        wandb_table=wandb_table,
        n_iter=n_iter,
        str_sfs="PB_Growing",
        bool_use_wandb=bool_use_wandb,
        bool_use_lightweight_wandb=bool_use_lightweight_wandb,
        bool_use_ray=True,
    )

    return vec_output_sfs, wandb_table


def get_pb_grow_features(
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
):
    # add breaks based on transition between channels
    idx_break = np.unique(
        np.concatenate([idx_break, np.where(np.diff(idx_ch) == 1)[0]])
    )
    str_metric_sfs = "avg_acc" if bool_force_acc else str_metric

    # now obtain all possible power bands to run analysis and form features
    vec_pb_full, vec_idx_peak_pb = group_pb_cross_asym(
        vec_output,
        features,
        idx_used,
        idx_break,
        str_metric=str_metric_sfs,
        max_width=max_width,
        width=width,
        top_k=n_candidate_peak,
    )
    vec_features_pb_sub = [
        correct_pb_feature_dim(features, pb, idx_used, n_iter) for pb in vec_pb_full
    ]

    return vec_features_pb_sub, vec_pb_full, vec_idx_peak_pb


def pb_grow_wrapper(*args, **kwargs):
    # run parallelized version of PB growing algorithm
    if kwargs["bool_use_ray"]:
        vec_output_pb_grow, wandb_pb_grow = sfs_pb_sweep_ray(
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

    # run serial version of feature PB growing algorithm
    else:
        vec_output_pb_grow, wandb_pb_grow = sfs_pb_sweep(
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

    return vec_output_pb_grow, wandb_pb_grow
