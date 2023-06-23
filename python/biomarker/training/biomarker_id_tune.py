import copy

import numpy as np
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from biomarker.training.model_initialize import get_model_params
from biomarker.training.seq_forward_selection import seq_forward_selection
from biomarker.training.biomarker_id import SFSTrainer


@hydra.main(version_base=None, config_path="../../conf", config_name="config_tune")
def biomarker_id_tune_sfs(
    cfg: DictConfig,
    **kwargs,
):
    # now initalize the trainer
    trainer = SFSTuneTrainer(cfg)
    trainer.train_side_tune()


def sfs_inner_loop_trainable(
    config: dict,
    features=None,
    y_class=None,
    y_stim=None,
    labels_cell=None,
    model_cfg_in: DictConfig | None = None,
    trainer_cfg_in: DictConfig | None = None,
    cfg: dict | None = None,
):
    # unpack the configs
    model_cfg = OmegaConf.to_container(model_cfg_in, resolve=True)
    trainer_cfg = OmegaConf.to_container(trainer_cfg_in, resolve=True)
    
    # for all the config keys, update the model_cfg
    for key, val in config.items():
        # update values in model_kwargs
        if key in model_cfg["model_kwargs"].keys():
            model_cfg["model_kwargs"][key] = val

        # update values in ensemble_kwargs
        if key in model_cfg["ensemble_kwargs"].keys():
            model_cfg["ensemble_kwargs"][key] = val
            
    # initialize wandb
    wandb = setup_wandb(
        config=config,
        project="SFS_{}_Tune".format(cfg["meta"]["str_model"]),
        group=cfg['tune']["wandb_group"],
    )

    # run the SFS process
    _, output_init, _, _, _, _ = seq_forward_selection(
        features,
        y_class,
        y_stim,
        labels_cell,
        model_cfg,
        trainer_cfg,
        n_ch=cfg["preproc"]["n_ch"],
        str_model=cfg["meta"]["str_model"],
        str_metric = cfg["meta"]["str_metric"],
        n_candidate_peak = cfg["feature_selection"]["n_candidate_peak"],
        n_candidate_pb = cfg["feature_selection"]["n_candidate_pb"],
        width = cfg["feature_selection"]["width"],
        max_width = cfg["feature_selection"]["max_width"],
        bool_force_sfs_acc = cfg["feature_selection"]["bool_force_sfs_acc"],
        random_seed = cfg["feature_selection"]["random_seed"],
        n_fold = cfg["feature_selection"]["n_fold"],
        bool_use_strat_kfold = cfg["feature_selection"]["bool_use_strat_kfold"],
        bool_use_ray=False,
        bool_use_gpu=cfg["parallel"]["bool_use_gpu"],
        bool_tune_hyperparams=cfg["feature_selection"]["bool_tune_hyperparams"],
        bool_verbose=False,
    )

    # obtain the metric
    avg_acc = output_init["vec_acc"][0]
    avg_auc = output_init["vec_auc"][0]

    session.report({"avg_acc": avg_acc, "avg_auc": avg_auc})
    wandb.log({'summary': dict(avg_acc=avg_acc, avg_auc=avg_auc)})
    
    
class SFSTuneTrainer(SFSTrainer):
    def __init__(self, cfg) -> None:
        # intialize the parent class
        super().__init__(cfg)
        
        # obtain tune related params
        self.num_samples = cfg["tune"]["num_samples"]

        # Note: ray_parameters here will not be used and will be handled by the ray.tune instead

        # quick sanity checks for hyperparameter tuning
        assert self.n_rep == 1, "Need to perform one iteration only"
        assert self.n_candidate_pb == 1, "Need to identify one power band only"
        assert self.bool_tune_hyperparams, "Hyperparameter tuning is not enabled"

    def parse_tune_kwargs(self, kwargs: DictConfig):
        # initialize search space variable
        search_space = dict()

        # parse all keyword arguments
        for kwarg_key, kwarg_val in kwargs.items():
            # only parse dictionaries
            if isinstance(kwarg_val, DictConfig):
                if "bool_tune" in kwarg_val.keys():
                    if kwarg_val.bool_tune:
                        # if type is range_choice
                        if kwarg_val.tune_op == "range_choice":
                            search_space[kwarg_key] = tune.choice(
                                np.arange(
                                    kwarg_val.tune_range[0],
                                    kwarg_val.tune_range[1],
                                    kwarg_val.tune_range[2],
                                )
                            )
                        elif kwarg_val.tune_op == "range_grid_search":
                            search_space[kwarg_key] = tune.grid_search(
                                np.arange(
                                    kwarg_val.tune_range[0],
                                    kwarg_val.tune_range[1],
                                    kwarg_val.tune_range[2],
                                )
                            )
                        elif kwarg_val.tune_op == "choice":
                            search_space[kwarg_key] = tune.choice(kwarg_val.tune_range)
                        elif kwarg_val.tune_op == "grid_search":
                            search_space[kwarg_key] = tune.grid_search(
                                kwarg_val.tune_range
                            )
                        else:
                            raise NotImplementedError(
                                f"tune_op {kwarg_val.tune_op} is not implemented"
                            )

        return search_space

    def initialize_search_space(self, model_cfg):
        # TODO: include parsing to args as well
        # parse the model and ensemble kwargs
        search_space_model = self.parse_tune_kwargs(model_cfg["model_kwargs"])
        search_space_ensemble = self.parse_tune_kwargs(model_cfg["ensemble_kwargs"])

        return search_space_model | search_space_ensemble

    def initialize_tune_config(self):
        tune_config = tune.TuneConfig(num_samples=self.num_samples)

        return tune_config

    def initialize_run_config(self):
        run_config = air.RunConfig(
            callbacks=[
                WandbLoggerCallback(project="SFS_{}_Tune".format(self.str_model))
            ]
        )

        return run_config

    def train_side_tune(self):
        # log into wandb and initialize
        wandb.login()

        # runtime sanity checks
        assert not self.bool_use_dyna, "Dynamics should not be used for model comp SFS"

        # load the data
        data_hemi = self.load_data()

        # load the model configurations (could be changed later)
        model_cfg, trainer_cfg = get_model_params(
            self.str_model,
            bool_use_gpu=self.bool_use_gpu,
            n_gpu_per_process=self.n_gpu_per_process,
            bool_tune_hyperparams=self.bool_tune_hyperparams,
        )

        # obtain the features
        features, y_class, y_stim, labels_cell = self.extract_features(
            data_hemi,
            n_dynamics=0,
        )

        # obtain the different configs for ray.tune.Tuner
        search_space = self.initialize_search_space(model_cfg)
        tune_config = self.initialize_tune_config()

        # initialize the ray.tune.Tuner
        resources = (
            {"cpu": self.n_cpu_per_process, "gpu": self.n_gpu_per_process}
            if self.bool_use_gpu
            else {"cpu": self.n_cpu_per_process}
        )
        sfs_inner_loop_w_resources = tune.with_resources(
            sfs_inner_loop_trainable, resources=resources
        )
        tuner = tune.Tuner(
            tune.with_parameters(
                sfs_inner_loop_w_resources,
                features=features,
                y_class=y_class,
                y_stim=y_stim,
                labels_cell=labels_cell,
                model_cfg_in=model_cfg,
                trainer_cfg_in=trainer_cfg,
                cfg=self.cfg,
            ),
            param_space=search_space,
            tune_config=tune_config,
        )

        # now run the tuning
        results = tuner.fit()

        return results


if __name__ == "__main__":
    biomarker_id_tune_sfs()
