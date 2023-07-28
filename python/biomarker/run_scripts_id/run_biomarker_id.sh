# conda init bash
conda activate envTorch201

# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --config-name config
# HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --config-name config patient=RCS17L 
HYDRA_FULL_ERROR=1 python ../training/biomarker_id.py --config-name config patient=RCS17R