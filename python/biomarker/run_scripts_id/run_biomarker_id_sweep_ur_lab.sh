# conda init bash
conda activate envTorch201

# # ur of 3s
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS02R preproc=ur3s
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS08R preproc=ur3s
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS11L preproc=ur3s

# # ur of 4_5s
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS02R preproc=ur4_5s
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS08R preproc=ur4_5s
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --multirun --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS11L preproc=ur4_5s

HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS02R preproc=ur4_5s meta=med_RF_auc_default
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS02R preproc=ur4_5s meta=med_RNN_auc_default
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --config-name config_sweep_ur_model parallel=lab_ray_cpu_batch patient=RCS02R preproc=ur4_5s meta=med_SVM_auc_default