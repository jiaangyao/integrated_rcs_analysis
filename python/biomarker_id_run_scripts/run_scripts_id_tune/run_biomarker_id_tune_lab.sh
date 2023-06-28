# conda init bash
conda activate envTorch201

HYDRA_FULL_ERROR=1 python ../training/biomarker_id_tune.py --config-name config_tune parallel=lab_ray_cpu_tune meta=med_RNN_auc_default tune.num_samples=1