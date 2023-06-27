# conda init bash
conda activate envTorch201

# subsec dynamics sweep through model and update rate for RCS02R
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_dynamics_subsec_ur patient=RCS02R preproc=ur0_2s parallel=lab_ray_cpu_batch
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_dynamics_subsec_ur patient=RCS02R preproc=ur0_3s parallel=lab_ray_cpu_batch
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_dynamics_subsec_ur patient=RCS02R preproc=ur0_4s parallel=lab_ray_cpu_batch
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_dynamics_subsec_ur patient=RCS02R preproc=ur0_5s parallel=home_ray_cpu_batch