# pyright: reportPrivateImportUsage=false
from __future__ import print_function
import os
import pathlib
import pickle

import ray
import tqdm
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from biomarker.training.prepare_data import prepare_data
from biomarker.training.seq_forward_selection import seq_forward_selection
from utils.parse_datetime import parse_dt_w_tz

_VEC_STR_SUBJECT = ("RCS02", "RCS08", "RCS17")
_VEC_STR_SIDE = ("L", "R")
_VEC_LABEL_TYPE = "med"
_VEC_STR_METRIC = ("avg_auc", "avg_acc", "avg_f1")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def biomarker_id_train_sfs(
    cfg: DictConfig,
    **kwargs,
):
    # now initalize the trainer
    trainer = SFSTrainer(cfg)
    trainer.train_side()


class DefaultModelTrainer:
    def __init__(self, cfg) -> None:
        """
        unpack the parameters from the config
        """
        # append the entire configuration
        self.cfg = cfg

        # unpack the experiment related meta parameters
        self.str_model = cfg["meta"]["str_model"]
        self.label_type = cfg["meta"]["label_type"]
        self.str_metric = cfg["meta"]["str_metric"]
        self.bool_debug = cfg["meta"]["bool_debug"]
        self.bool_tune_hyperparams = cfg["meta"]["bool_tune_hyperparams"]

        # unpack the subject specfic configs
        self.str_subject = cfg["patient"]["str_subject"]
        self.str_side = cfg["patient"]["str_side"]
        self.stim_level = cfg["patient"]["stim_settings"][self.str_side]["amp_in_mA"]

        # unpack the paths to the data to be loaded
        self.p_data = pathlib.Path(cfg["data_paths"]["p_data"])
        self.f_data = cfg["data_paths"]["f_data"]
        self.p_output = pathlib.Path(cfg["data_paths"]["p_output"])
        self.f_output = cfg["data_paths"]["f_output"]

        # unpack the preprocessing related parameters
        self.n_ch = cfg["preproc"]["n_ch"]
        self.interval = cfg["preproc"]["interval"]
        self.update_rate = cfg["preproc"]["update_rate"]
        self.freq_low_lim = cfg["preproc"]["freq_low_lim"]
        self.freq_high_lim = cfg["preproc"]["freq_high_lim"]

        # unpack the parameters to the feature selection pipeline
        # will be instantiated later in subclasses

        # unapck the parallelization related parameters
        self.bool_use_ray = cfg["parallel"]["bool_use_ray"]
        self.bool_use_gpu = cfg["parallel"]["bool_use_gpu"]
        self.bool_use_batch = cfg["parallel"]["bool_use_batch"]
        self.n_cpu = cfg["parallel"]["n_cpu"]
        self.n_gpu = cfg["parallel"]["n_gpu"]
        self.n_cpu_per_process = cfg["parallel"]["n_cpu_per_process"]
        self.n_gpu_per_process = cfg["parallel"]["n_gpu_per_process"]
        self.batch_size = cfg["parallel"]["batch_size"]

        # unpack the dynamics related parameters
        self.bool_use_dyna = cfg["dynamics"]["bool_use_dyna"]
        self.n_dyna_start = cfg["dynamics"]["n_dyna_start"]
        self.n_dyna_stop = cfg["dynamics"]["n_dyna_stop"]

        # sanity checks
        assert (
            self.str_subject in _VEC_STR_SUBJECT
        ), "Data must be from defined subjects"
        assert (
            self.str_side in _VEC_STR_SIDE
        ), "Data must be from the left or right hemispheres"

    def train_side(self):
        raise NotImplementedError("This method should be overriden by the subclass")


class SFSTrainer(DefaultModelTrainer):
    def __init__(self, cfg) -> None:
        """
        unpack the parameters from the config
        """
        super().__init__(cfg)

        # unpack the parameters to the feature selection pipeline
        self.str_feature_selection = cfg["feature_selection"]["name"]

        # unpack the meta settings for feature selection

        self.n_rep = cfg["feature_selection"]["n_rep"]
        self.n_candidate_peak = cfg["feature_selection"]["n_candidate_peak"]
        self.n_candidate_pb = cfg["feature_selection"]["n_candidate_pb"]
        self.width = cfg["feature_selection"]["width"]
        self.max_width = cfg["feature_selection"]["max_width"]
        self.bool_force_sfs_acc = cfg["feature_selection"]["bool_force_sfs_acc"]

        self.n_fold = cfg["feature_selection"]["n_fold"]
        self.bool_use_strat_kfold = cfg["feature_selection"]["bool_use_strat_kfold"]
        self.random_seed = cfg["feature_selection"]["random_seed"]

        # sanity checks
        assert self.label_type in _VEC_LABEL_TYPE, "Label type must be defined"
        assert self.str_metric in _VEC_STR_METRIC, "Metric must be defined"
        assert not (self.str_metric == "avg_acc" and not self.bool_force_sfs_acc)
        assert not (self.str_metric != "avg_acc" and self.bool_force_sfs_acc)

        # set the right number of CPUs and GPUs to use
        if self.bool_use_ray:
            # first update according to batch size
            n_band_max = self.n_ch * (self.freq_high_lim - self.freq_low_lim + 1)
            self.n_cpu_per_process = (
                np.ceil(self.n_cpu / np.ceil(n_band_max / self.batch_size))
                if self.bool_use_batch
                else self.n_cpu_per_process
            )

            self.n_gpu_per_process = (
                0.8 * self.n_gpu / np.round(self.n_cpu / self.n_cpu_per_process)
                if self.bool_use_batch and self.bool_use_gpu
                else self.n_gpu_per_process
            )

            # set the GPU per process
            self.n_cpu_per_process = (
                np.ceil(self.n_cpu / np.ceil(self.n_gpu / self.n_gpu_per_process))
                if self.bool_use_gpu
                else self.n_cpu_per_process
            )

        # debug related flags
        if self.bool_debug:
            Warning("Debug mode is on, only 1 rep will be run")
            self.n_rep = 1

    def load_data(self):
        # load the data from the desginated side
        data_hemi = pd.read_csv(str(self.p_data / self.f_data), header=0)

        # quickly convert the time to pd.Timestamp
        data_hemi["time"] = parse_dt_w_tz(
            data_hemi["time"],
            dt_fmt="%d-%b-%Y %H:%M:%S.%f",
            tz_str="America/Los_Angeles",
        )

        return data_hemi

    def initialize_ray(self):
        # initialize ray
        if self.bool_use_ray:
            if not self.bool_use_gpu:
                os.environ["RAY_DEDUP_LOGS"] = "0"
                context = ray.init(
                    log_to_driver=False,
                    num_cpus=self.n_cpu,
                    include_dashboard=True,
                )
                # ray.init(ignore_reinit_error=True, logging_level=40, include_dashboard=True)

            else:
                # initailize ray with GPU
                # ptu.init_gpu(use_gpu=True, bool_use_best_gpu=True)
                # os.environ["CUDA_VISIBLE_DEVICES"] = str(ptu.device.index)
                os.environ["RAY_DEDUP_LOGS"] = "0"
                context = ray.init(
                    log_to_driver=False,
                    num_cpus=self.n_cpu,
                    num_gpus=self.n_gpu,
                    include_dashboard=True,
                )

        else:
            context = None

        return context

    def terminate_ray(self, context):
        # terminate ray
        if self.bool_use_ray:
            ray.shutdown()

    def extract_features(
        self,
        data_hemi,
        n_dynamics=0,
    ):
        # obtain the features
        features, y_class, y_stim, labels_cell, _ = prepare_data(
            data_hemi,
            str_side=self.str_side,
            stim_level=self.stim_level,
            label_type=self.label_type,
            interval=self.interval,
            update_rate=self.update_rate,
            low_lim=self.freq_low_lim,
            high_lim=self.freq_high_lim,
            bool_use_dyna=self.bool_use_dyna,
            n_dynamics=n_dynamics,
        )

        return features, y_class, y_stim, labels_cell

    def save_output(
        self,
        output: dict,
    ):
        """save the output file

        Args:
            output (dict): dictionary containing the output
        """
        # append config to output
        output["cfg"] = self.cfg

        # confirm the output directory exists
        self.p_output.mkdir(parents=True, exist_ok=True)

        # dump output
        with open(str(self.p_output / self.f_output), "wb") as f:
            pickle.dump(output, f)

    def SFS_inner_loop(
        self,
        features,
        y_class,
        y_stim,
        labels_cell,
    ):
        # iterate through the repetitions
        print("\nSequential Forward Selection")
        for idx_rep in tqdm.trange(
            self.n_rep,
            leave=False,
            desc="{} REP".format(self.str_feature_selection),
            bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        ):
            # initialize the output variable
            output_label = dict()
            output_label["sinPB"] = []
            output_label["sfsPB"] = []

            # perform the SFS
            output_fin, output_init, _, _ = seq_forward_selection(
                features,
                y_class,
                y_stim,
                labels_cell,
                n_ch=self.n_ch,
                str_model=self.str_model,
                str_metric=self.str_metric,
                n_candidate_peak=self.n_candidate_peak,
                n_candidate_pb=self.n_candidate_pb,
                width=self.width,
                max_width=self.max_width,
                bool_force_sfs_acc=self.bool_force_sfs_acc,
                random_seed=self.random_seed,
                n_fold=self.n_fold,
                bool_use_strat_kfold=self.bool_use_strat_kfold,
                bool_use_ray=self.bool_use_ray,
                bool_use_gpu=self.bool_use_gpu,
                n_cpu_per_process=self.n_cpu_per_process,
                n_gpu_per_process=self.n_gpu_per_process,
                bool_use_batch=self.bool_use_batch,
                batch_size=self.batch_size,
                bool_tune_hyperparams=self.bool_tune_hyperparams,
            )

            # append to outer list
            output_label["sinPB"].append(output_init)
            output_label["sfsPB"].append(output_fin)
            print("\nHighest SinPB auc: {:.4f}".format(output_init["vec_auc"][0]))
            print("Highest SFS auc: {:.4f}".format(output_fin["vec_auc"][-1]))
            print("Done with rep {}".format(idx_rep + 1))
            print("")

        # return the output
        return output_label

    def train_side(self):
        """Training pipeline for a single side of the brain"""
        
        # runtime sanity checks
        assert not self.bool_use_dyna, "Dynamics should not be used for model comp SFS"

        # load the data
        data_hemi = self.load_data()

        # initialize ray
        context = self.initialize_ray()

        # obtain the features
        features, y_class, y_stim, labels_cell = self.extract_features(
            data_hemi,
            n_dynamics=0,
        )

        # perform SFS inner loop and iterate through the repetitions
        output_label = self.SFS_inner_loop(
            features,
            y_class,
            y_stim,
            labels_cell,
        )

        # shutdown ray in case of using it
        self.terminate_ray(context)

        # save the output
        self.save_output(output_label)

        # final print statement for breakpoint
        if self.bool_debug:
            print("Debug breakpoint")


if __name__ == "__main__":
    biomarker_id_train_sfs()
    # raise RuntimeError(
    #     "Cannot run this script directly, must be called by separate run scripts"
    # )
