from __future__ import print_function
import pathlib
import pickle
import typing as tp
from types import MappingProxyType

import ray
import tqdm
import pandas as pd

from biomarker.training.prepare_data import prepare_data
from biomarker.training.seq_forward_selection import seq_forward_selection
from utils.parse_datetime import parse_dt_w_tz


_VEC_STR_SUBJECT = tp.Literal["RCS02"]
_VEC_STR_SIDE = tp.Literal["L", "R"]
_VEC_MED_LEVEL = tp.Literal["med"]
_VEC_STR_METRIC = tp.Literal["avg_auc", "avg_acc", "avg_f1"]


def gen_config(
    p_data: pathlib.Path,
    f_data: str,
    p_output: pathlib.Path,
    f_output_opt: str = "",
    str_subject: _VEC_STR_SUBJECT = "RCS02",
    str_side: _VEC_STR_SIDE = "R",
    stim_level: dict | MappingProxyType = MappingProxyType(dict()),
    label_type: _VEC_MED_LEVEL = "med",
    str_model: str = "LDA",
    str_metric: _VEC_STR_METRIC = "avg_auc",
    interval: float = 0.05,
    update_rate: int = 30,
    freq_low_lim: int = 2,
    freq_high_lim: int = 100,
    n_rep: int = 5,
    bool_debug: bool = False,
    bool_use_ray: bool = True,
    bool_use_gpu: bool = False,
    bool_force_sfs_acc: bool = False,
    bool_use_strat_kfold: bool = True,
    random_seed: int | None = None,
) -> dict:
    """Creates the default dictionary for choosing the various hyperparameters for setting up training

    Args:
        p_data (pathlib.Path): Absolute path to the directory holding all data
        f_data (str): Name of the file holding the data w/. extension
        p_output (pathlib.Path): Absolute path to the directory holding all outputs
        f_output_opt (str): Optional extension to the output file name. Defaults to ''.
        str_subject (_VEC_STR_SUBJECT, optional): Name of the subject. Defaults to 'RCS02'.
        str_side (_VEC_STR_SIDE, optional): Name of the hemisphere. Defaults to 'R'.
        stim_level (dict | MappingProxyType, optional): Dictionary holding the stimulation level. Defaults to MappingProxyType(dict()).
        label_type (_VEC_MED_LEVEL, optional): Type of the output label to run classification on. Defaults to 'med'.
        interval (float, optional): Skipping interval of FFT windows. Defaults to 0.05s (50ms).
        update_rate (int, optional): Number of FFT windows to smooth over to form power output. Defaults to 30.
        freq_low_lim (int, optional): Lower frequency limit for the SFS search. Defaults to 2Hz.
        freq_high_lim (int, optional): Upper frequency limit for the SFS search. Defaults to 100Hz.
        n_rep (int, optional): Number of repetitions. Defaults to 5.
        bool_debug (bool, optional): Boolean flag for debug mode. Defaults to False.
        bool_use_ray (bool, optional): Boolean flag for using ray to parallelize the code. Defaults to True.
        bool_force_sfs_acc (bool, optional): Boolean flag for forcing to use acc as metric. Defaults to False.
        bool_use_strat_kfold (bool, optional): Boolean flag for using stratified k-fold. Defaults to True.
        str_model (str, optional): Name of the model. Defaults to 'LDA'.
        str_metric (str, optional): Name of the metric. Defaults to 'avg_auc'.
        random_seed (int | None, optional): Random seed for all the cross validation. Defaults to None.

    Returns:
        dict: Dictionary holding all the configuration parameters for the biomarker identification
    """

    # create the empty dictionary
    cfg = dict()

    """
    now get the various parameters into the config dictionary
    """

    # starting with the path variables
    cfg["p_data"] = p_data
    cfg["f_data"] = f_data
    cfg["p_output"] = p_output
    cfg["f_output_opt"] = f_output_opt

    # next unpack the subject specific configs
    cfg["str_subject"] = str_subject
    cfg["str_side"] = str_side
    cfg["stim_level"] = stim_level
    cfg["label_type"] = label_type

    # unpack the preprocessing related parameters
    cfg["interval"] = interval
    cfg["update_rate"] = update_rate
    cfg["freq_low_lim"] = freq_low_lim
    cfg["freq_high_lim"] = freq_high_lim

    # unpack the SFS training related flags
    cfg["n_rep"] = n_rep
    cfg["bool_debug"] = bool_debug
    cfg["bool_use_ray"] = bool_use_ray
    cfg["bool_use_gpu"] = bool_use_gpu
    cfg["bool_force_sfs_acc"] = bool_force_sfs_acc
    cfg["bool_use_strat_kfold"] = bool_use_strat_kfold
    cfg["str_model"] = str_model
    cfg["str_metric"] = str_metric
    cfg["random_seed"] = random_seed
    assert not (str_metric == "avg_acc" and not bool_force_sfs_acc)
    assert not (str_metric == "avg_acc" and bool_force_sfs_acc)

    return cfg


class BiomarkerIDTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def train_side(self):
        """Biomarker identification script

        Args:
            cfg (dict): dictionary holding the configurations for the biomarker identification
        """

        # obtain the config dictionary
        cfg = self.cfg

        # unpack the path variables
        p_data = cfg["p_data"]
        f_data = cfg["f_data"]
        p_output = cfg["p_output"]
        f_output_opt = cfg["f_output_opt"]

        # unpack the subject specfic configs
        str_subject = cfg["str_subject"]
        str_side = cfg["str_side"]
        stim_level = cfg["stim_level"]
        label_type = cfg["label_type"]
        assert str_side in ["L", "R"], "Data must be from the left or right hemispheres"
        if label_type not in ["med"]:
            raise NotImplementedError("Currently only med labels are supported")

        # unpack the preprocessing related parameters
        interval = cfg["interval"]
        update_rate = cfg["update_rate"]
        freq_low_lim = cfg["freq_low_lim"]
        freq_high_lim = cfg["freq_high_lim"]

        # unpack the SFS training related flags
        n_rep = cfg["n_rep"]
        bool_debug = cfg["bool_debug"]
        bool_use_ray = cfg["bool_use_ray"]
        bool_use_gpu = cfg["bool_use_gpu"]
        bool_force_sfs_acc = cfg["bool_force_sfs_acc"]
        bool_use_strat_kfold = cfg["bool_use_strat_kfold"]
        str_model = cfg["str_model"]
        str_metric = cfg["str_metric"]
        random_seed = cfg["random_seed"]
        assert not (str_metric == "avg_acc" and not bool_force_sfs_acc)
        assert not (str_metric == "avg_acc" and bool_force_sfs_acc)
        if bool_debug:
            Warning("Debug mode is on, only 1 rep will be run")
            n_rep = 1

        # load the data from the desginated side
        data_hemi = pd.read_csv(str(p_data / f_data), header=0)

        # quickly convert the time to pd.Timestamp
        data_hemi["time"] = parse_dt_w_tz(
            data_hemi["time"],
            dt_fmt="%d-%b-%Y %H:%M:%S.%f",
            tz_str="America/Los_Angeles",
        )

        # initialize ray
        # ray.init(ignore_reinit_error=True, logging_level=40, include_dashboard=True)
        if bool_use_ray:
            if not bool_use_gpu:
                context = ray.init(log_to_driver=False)
                print(context.dashboard_url)  # type: ignore
            else:
                raise NotImplementedError("GPU support not yet implemented")
                ray.init(log_to_driver=False, num_gpus=1)

        # initialize the output variable
        output_label = dict()
        output_label["sinPB"] = []
        output_label["sfsPB"] = []

        # iterate through the repetitions
        for idx_rep in tqdm.trange(
            n_rep,
            leave=False,
            desc="SFS REP",
            bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        ):
            # obtain the features
            features, y_class, y_stim, labels_cell, _ = prepare_data(
                data_hemi,
                stim_level,
                str_side=str_side,
                label_type=label_type,
                interval=interval,
                update_rate=update_rate,
                low_lim=freq_low_lim,
                high_lim=freq_high_lim,
            )

            # perform the SFS
            output_fin, output_init, iter_used, orig_metric = seq_forward_selection(
                features,
                y_class,
                y_stim,
                labels_cell,
                str_model=str_model,
                bool_force_sfs_acc=bool_force_sfs_acc,
                bool_use_ray=bool_use_ray,
                bool_use_strat_kfold=bool_use_strat_kfold,
                random_seed=random_seed,
            )

            # append to outer list
            output_label["sinPB"].append(output_init)
            output_label["sfsPB"].append(output_fin)
            print("\nHighest SinPB auc: {:.4f}".format(output_init["vec_auc"][0]))
            print("Highest SFS auc: {:.4f}".format(output_fin["vec_auc"][-1]))
            print("Done with rep {}".format(idx_rep + 1))
            print("")

        # shutdown ray in case of using it
        if bool_use_ray:
            ray.shutdown()

        # save the output file
        output_label["cfg"] = cfg
        f_output = "{}_{}_{}_{}_{}{}.pkl".format(
            str_subject, str_side, label_type, str_metric, str_model, f_output_opt
        )
        with open(str(p_output / f_output), "wb") as f:
            pickle.dump(output_label, f)

        # final print statement for breakpoint
        if bool_debug:
            print("Debug breakpoint")


if __name__ == "__main__":
    raise RuntimeError(
        "Cannot run this script directly, must be called by separate run scripts"
    )
