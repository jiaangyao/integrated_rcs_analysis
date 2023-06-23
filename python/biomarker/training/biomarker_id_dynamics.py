from __future__ import print_function

import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

from biomarker.training.biomarker_id import SFSTrainer


_VEC_STR_SUBJECT = ("RCS02", "RCS08", "RCS17")
_VEC_STR_SIDE = ("L", "R")
_VEC_LABEL_TYPE = "med"
_VEC_STR_METRIC = ("avg_auc", "avg_acc", "avg_f1")


@hydra.main(version_base=None, config_path="../../conf", config_name="config_dynamics")
def dynamics_train_sfs(
    cfg: DictConfig,
    **kwargs,
):
    """Training script for dynamics model for single side of the brain"""

    # intialize the trainer
    trainer = SFSDynamicsTrainer(cfg)

    # train the model
    trainer.train_side_dyna()


class SFSDynamicsTrainer(SFSTrainer):
    def __init__(self, cfg) -> None:
        # intialize the parent class
        super().__init__(cfg)

        # quick sanity checks for dynamics-based analysis
        assert self.bool_use_dyna, "Dynamics-based analysis is not enabled"

    def train_side_dyna(self):
        """Training script for dynamics model for single side of the brain"""

        # load the data
        data_hemi = self.load_data()

        # initialize ray
        context = self.initialize_ray()

        # initialize the output variable
        output_full_level = dict()
        output_full_level["sinPB"] = dict()
        output_full_level["sfsPB"] = dict()

        # loop through the various dynamics lengths first
        for n_dynamics in tqdm.trange(
            self.n_dyna_start,
            self.n_dyna_stop,
            leave=False,
            desc="N DYNA",
            bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}",
        ):
            # setting up header for current iteration
            print("\n======================================================")
            print(
                "N DYNAMICS: {}, ABS DYNAMICS: {:.3f}\n".format(
                    n_dynamics, n_dynamics * self.update_rate * self.interval
                )
            )

            # obtain the features
            features, y_class, y_stim, labels_cell = self.extract_features(
                data_hemi,
                n_dynamics=n_dynamics,
            )

            # perform SFS inner loop and iterate through the repetitions
            output_med_level = self.SFS_inner_loop(
                features,
                y_class,
                y_stim,
                labels_cell,
            )

            # append to outer list
            # initialize the dictionary if not already done
            if (
                "n_dynamics_{}".format(n_dynamics)
                not in output_full_level["sinPB"].keys()
            ):
                output_full_level["sinPB"]["n_dynamics_{}".format(n_dynamics)] = []
                output_full_level["sfsPB"]["n_dynamics_{}".format(n_dynamics)] = []
            output_full_level["sinPB"]["n_dynamics_{}".format(n_dynamics)].append(
                output_med_level["sinPB"]
            )
            output_full_level["sfsPB"]["n_dynamics_{}".format(n_dynamics)].append(
                output_med_level["sfsPB"]
            )

        # shutdown ray in case of using it
        self.terminate_ray(context)

        # save the output
        self.save_output(output_full_level)

        # final print statement for breakpoint
        if self.bool_debug:
            print("Debug breakpoint")


if __name__ == "__main__":
    dynamics_train_sfs()
    # raise RuntimeError(
    #     "Cannot run this script directly, must be called by separate run scripts"
    # )
