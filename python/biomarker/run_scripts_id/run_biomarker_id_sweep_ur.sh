# conda init bash
conda activate envTorch201

# UR of 6s
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=home_ray_cpu_batch patient=RCS02R preproc=ur6s
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=home_ray_cpu_batch patient=RCS08R preproc=ur6s
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=home_ray_cpu_batch patient=RCS11L preproc=ur6s