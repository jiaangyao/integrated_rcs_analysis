# pyright: reportPrivateImportUsage=false
import copy

import tqdm
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from omegaconf import DictConfig

from .base import get_idx_ch, _MAX_NUMBER_OF_FINAL_PB
from biomarker.PBgen.hymer_pb_wrapper import hymer_pb_method
from biomarker.PBgrow.sfs import sfs_forward_pass


def run_pb_pipeline(
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
    # define initial variables
    idx_ch = get_idx_ch(n_ch, features.shape[1])
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

    # create variables to be returned later
    orig_metric = None
    model_cfg_pb = None
    trainer_cfg_pb = None

    # initialize table for PB filtering and PB growing if using wandb
    wandb_pb_filt = None
    wandb_pb_grow = None

    # start the main loop
    for n_iter in tqdm.trange(
        1,
        n_fin_pb + 1,
        leave=False,
        desc="ITER",
        bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        disable=not bool_verbose,
    ):
        # now run the HJ PB method to identify the candidate PBs for current iteration
        (
            vec_output_pb_grow,
            vec_pb_full,
            vec_idx_peak_pb,
            model_cfg_pb,
            trainer_cfg_pb,
            wandb_pb_filt,
            wandb_pb_grow,
            orig_metric,
        ) = hymer_pb_method(
            features,
            y_class,
            y_stim,
            idx_ch,
            idx_all,
            idx_used,
            bool_used,
            n_iter,
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            wandb_pb_filt=wandb_pb_filt,
            wandb_pb_grow=wandb_pb_grow,
            str_model=str_model,
            str_metric=str_metric,
            n_candidate_peak=n_candidate_peak,
            width=width,
            max_width=max_width,
            bool_force_acc=bool_force_acc,
            random_seed=random_seed,
            n_fold=n_fold,
            bool_use_strat_kfold=bool_use_strat_kfold,
            bool_use_ray=bool_use_ray,
            bool_use_gpu=bool_use_gpu,
            n_cpu_per_process=n_cpu_per_process,
            n_gpu_per_process=n_gpu_per_process,
            bool_use_batch=bool_use_batch,
            batch_size=batch_size,
            bool_tune_hyperparams=bool_tune_hyperparams,
            bool_use_wandb=bool_use_wandb,
            bool_use_lightweight_wandb=bool_use_lightweight_wandb,
            bool_verbose=bool_verbose,
        )

        # next run SFS forward pass to select the PB that would be used for this round
        output_fin, output_init, vec_output_sfs = sfs_forward_pass(
            output_fin,
            output_init,
            vec_output_pb_grow,
            vec_pb_full,
            vec_idx_peak_pb,
            idx_all,
            str_metric,
            n_fin_pb,
            n_candidate_pb,
        )

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
    for i in range(min([_MAX_NUMBER_OF_FINAL_PB, len(output_fin["vec_pb_ord"])])):
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
    for i in range(min([_MAX_NUMBER_OF_FINAL_PB, len(output_init["vec_pb_ord"])])):
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
        _MAX_NUMBER_OF_FINAL_PB if n_fin_pb > _MAX_NUMBER_OF_FINAL_PB else n_fin_pb
    )  # force most to be _MAX_NUMBER_OF_FINAL_PB
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

    # sanity check for missing variables
    assert orig_metric is not None, "Original metric from first iteration is None!"
    assert model_cfg_pb is not None, "Model config loaded from latest run is None!"
    assert trainer_cfg_pb is not None, "Trainer config loaded from latest run is None!"

    # return the outputs
    return output_fin, output_init, iter_used, orig_metric, model_cfg_pb, trainer_cfg_pb
