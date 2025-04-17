# conda init bash
conda activate envTorch201

# run the hydra training script
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sleep parallel=home_ray_cpu_batch patient=RCS02L