# conda init bash
conda activate envTorch201

HYDRA_FULL_ERROR=1 python ../training/biomarker_id_tune.py --config-name config_tune parallel=lab_ray_cpu tune.num_samples=240