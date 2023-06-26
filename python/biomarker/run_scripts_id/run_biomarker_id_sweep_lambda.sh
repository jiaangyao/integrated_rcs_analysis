# conda init bash
conda activate envTorch201

HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_model parallel=lambda_ray_cpu_batch patient=RCS02R
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_model parallel=lambda_ray_cpu_batch patient=RCS08R
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_model parallel=lambda_ray_cpu_batch patient=RCS11L
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_model parallel=lambda_ray_cpu_batch patient=RCS18L
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_model parallel=lambda_ray_cpu_batch patient=RCS12L