# conda init bash
conda activate envTorch201

# subsec dynamics sweep through model for RCS02R
HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_subsec patient=RCS02R parallel=home_ray_cpu_batch 