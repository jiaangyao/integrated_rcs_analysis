# conda init bash
conda activate envTorch201

# HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model
HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model parallel=home_ray_cpu_batch patient=RCS11L
HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model parallel=home_ray_cpu_batch patient=RCS12L
HYDRA_FULL_ERROR=1 python ../training/biomarker_id_dynamics.py --multirun --config-name config_dynamics_sweep_model parallel=home_ray_cpu_batch patient=RCS18L