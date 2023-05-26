import pathlib

from biomarker.run_scripts_id.biomarker_id import gen_config, BiomarkerIDTrainer


if __name__ == "__main__":
    # hardcode path to the RCS02 step 3 data for now
    p_project = pathlib.Path("/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/")
    p_data = p_project / "Data/RCS02/Step3_in_clinic_neural_recordings/"
    p_output = pathlib.Path("/home/jyao/Downloads/biomarker_id/model_id/")
    f_data_L = "rcs02_L_table.csv"
    f_data_R = "rcs02_R_table.csv"
    stim_level = dict()
    stim_level["L"] = [1.7, 2.5]
    stim_level["R"] = [3, 3.4]

    # # generate the config (UR=1.5s)
    # cfg = gen_config(
    #     p_data,
    #     f_data_R,
    #     p_output,
    #     str_subject="RCS02",
    #     str_side="R",
    #     stim_level=stim_level,
    #     label_type="med",
    #     str_model="LDA",
    #     str_metric="avg_auc",
    #     interval=0.05,
    #     update_rate=30,
    #     n_rep=5,
    #     bool_debug=False,
    #     bool_use_ray=True,
    #     bool_use_gpu=False,
    #     bool_force_sfs_acc=False,
    #     bool_use_strat_kfold=False,
    #     random_seed=None,
    # )

    # # generate the config (UR=3s)
    # cfg = gen_config(
    #     p_data,
    #     f_data_R,
    #     p_output,
    #     f_output_opt="_UR60",
    #     str_subject="RCS02",
    #     str_side="R",
    #     stim_level=stim_level,
    #     label_type="med",
    #     str_model="LDA",
    #     str_metric="avg_auc",
    #     interval=0.05,
    #     update_rate=60,
    #     n_rep=5,
    #     bool_debug=False,
    #     bool_use_ray=True,
    #     bool_use_gpu=False,
    #     bool_force_sfs_acc=False,
    #     bool_use_strat_kfold=False,
    #     random_seed=None,
    # )

    # # generate the config (UR=4.5s)
    # cfg = gen_config(
    #     p_data,
    #     f_data_R,
    #     p_output,
    #     f_output_opt="_UR90",
    #     str_subject="RCS02",
    #     str_side="R",
    #     stim_level=stim_level,
    #     label_type="med",
    #     str_model="LDA",
    #     str_metric="avg_auc",
    #     interval=0.05,
    #     update_rate=90,
    #     n_rep=5,
    #     bool_debug=False,
    #     bool_use_ray=True,
    #     bool_use_gpu=False,
    #     bool_force_sfs_acc=False,
    #     bool_use_strat_kfold=False,
    #     random_seed=None,
    # )

    # generate the config (UR=6s)
    # cfg = gen_config(
    #     p_data,
    #     f_data_R,
    #     p_output,
    #     f_output_opt="_UR120",
    #     str_subject="RCS02",
    #     str_side="R",
    #     stim_level=stim_level,
    #     label_type="med",
    #     str_model="LDA",
    #     str_metric="avg_auc",
    #     interval=0.05,
    #     update_rate=120,
    #     n_rep=5,
    #     bool_debug=False,
    #     bool_use_ray=True,
    #     bool_use_gpu=False,
    #     bool_force_sfs_acc=False,
    #     bool_use_strat_kfold=False,
    #     random_seed=None,
    # )

    # # generate the config (UR=7.5s)
    # cfg = gen_config(
    #     p_data,
    #     f_data_R,
    #     p_output,
    #     f_output_opt="_UR150",
    #     str_subject="RCS02",
    #     str_side="R",
    #     stim_level=stim_level,
    #     label_type="med",
    #     str_model="LDA",
    #     str_metric="avg_auc",
    #     interval=0.05,
    #     update_rate=150,
    #     n_rep=5,
    #     bool_debug=False,
    #     bool_use_ray=True,
    #     bool_use_gpu=False,
    #     bool_force_sfs_acc=False,
    #     bool_use_strat_kfold=False,
    #     random_seed=None,
    # )

    # generate the config (UR=9s)
    cfg = gen_config(
        p_data,
        f_data_R,
        p_output,
        f_output_opt="_UR180",
        str_subject="RCS02",
        str_side="R",
        stim_level=stim_level,
        label_type="med",
        str_model="LDA",
        str_metric="avg_auc",
        interval=0.05,
        update_rate=180,
        n_rep=5,
        bool_debug=False,
        bool_use_ray=True,
        bool_use_gpu=False,
        bool_force_sfs_acc=False,
        bool_use_strat_kfold=False,
        random_seed=None,
    )

    # initialize training
    trainer = BiomarkerIDTrainer(cfg)
    trainer.train_side()
