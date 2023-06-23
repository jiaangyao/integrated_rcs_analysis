# conda init bash
conda activate envTorch201

HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_tune parallel=lambda_ray_cpu_batch