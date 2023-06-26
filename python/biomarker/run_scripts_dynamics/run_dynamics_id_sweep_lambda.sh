# conda init bash
conda activate envTorch201

HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model parallel=lambda_ray_cpu_batch patient=RCS02R
HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model parallel=lambda_ray_cpu_batch patient=RCS08R
HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model parallel=lambda_ray_cpu_batch patient=RCS11L