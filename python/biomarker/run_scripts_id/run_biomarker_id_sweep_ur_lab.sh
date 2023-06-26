# conda init bash
conda activate envTorch201

# ur of 3s
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS02R preproc=ur3s
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS08R preproc=ur3s
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS11L preproc=ur3s