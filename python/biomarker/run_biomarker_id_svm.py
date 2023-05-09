import pathlib
import pickle

import ray
import pandas as pd
from easydict import EasyDict as edict

from biomarker.training.prepare_data import prepare_data
from biomarker.training.seq_forward_selection import seq_forward_selection
from utils.parse_datetime import parse_dt_w_tz

# hardcode path to the RCS02 step 3 data for now
p_project = pathlib.Path('/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/')
p_data = p_project / 'Data/RCS02/Step3_in_clinic_neural_recordings/'
p_output = pathlib.Path('/home/jyao/Downloads/')
f_data_L = 'rcs02_L_table.csv'
f_data_R = 'rcs02_R_table.csv'

# load the data into python
data_R = pd.read_csv(str(p_data / f_data_R), header=0)

# quickly convert the time to pd.Timestamp
data_R['time'] = parse_dt_w_tz(data_R['time'], dt_fmt='%d-%b-%Y %H:%M:%S.%f', tz_str='America/Los_Angeles')

# initialize ray
# ray.init(ignore_reinit_error=True, logging_level=40, include_dashboard=True)
ray.init(log_to_driver=False)

# now set out to perform cv-based biomarker identification
stim_level = edict(); stim_level.L = [1.7, 2.5]; stim_level.R = [3, 3.4]
output_med_level = edict(); output_med_level.sinPB = []; output_med_level.sfsPB = []
for idx_rep in range(5):
    print('\nrep {}'.format(idx_rep + 1))

    # obtain the features
    features, y_class, y_stim, labels_cell, _ = prepare_data(data_R, stim_level, str_side='R', label_type='med')

    # perform the SFS
    output_fin, output_init, iter_used, orig_metric = seq_forward_selection(features, y_class, y_stim, labels_cell,
                                                                            str_model='SVM', random_seed=None,
                                                                            bool_force_sfs_acc=False)

    # append to outer list
    output_med_level.sinPB.append(output_init)
    output_med_level.sfsPB.append(output_fin)
    print('\nHighest SinPB auc: {:.4f}'.format(output_init['vec_auc'][0]))
    print('Highest SFS auc: {:.4f}'.format(output_fin['vec_auc'][-1]))
    print('Done with rep {}'.format(idx_rep + 1))


# save the output
with open(str(p_output / 'RCS02_R_med_level_auc_SVM.pkl'), 'wb') as f:
    pickle.dump(output_med_level, f)

# shutdown ray
ray.shutdown()

print('debug')
