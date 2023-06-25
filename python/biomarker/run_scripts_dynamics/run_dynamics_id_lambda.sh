# conda init bash
conda activate envTorch201

HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --config-name config_dynamics parallel=lambda_ray_cpu_batch