import pathlib
import pickle

import tqdm
import ray
import pandas as pd
from easydict import EasyDict as edict

from biomarker.training.prepare_data import prepare_data
from biomarker.training.seq_forward_selection import seq_forward_selection
from utils.parse_datetime import parse_dt_w_tz

# hardcode path to the RCS02 step 3 data for now
p_project = pathlib.Path('/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/')
p_data = p_project / 'Data/RCS02/Step3_in_clinic_neural_recordings/'
p_output = pathlib.Path('/home/jyao/Downloads/biomarker_id/dynamics/')
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
stim_level = dict(); stim_level['L'] = [1.7, 2.5]; stim_level['R'] = [3, 3.4]
output_full_level = dict(); output_full_level['sinPB'] = {}; output_full_level['sfsPB'] = {}

# loop through the signal dynamics first
for n_dynamics in tqdm.trange(1, 7, leave=False,desc='N DYNA', bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}"):

    # setting up the variable for storing output
    print('\n=====================================')
    print('LENGTH OF DYNAMICS: {}\n'.format(n_dynamics))
    output_med_level = dict()
    output_med_level['sinPB'] = []
    output_med_level['sfsPB'] = []

    # obtain the features
    features, y_class, y_stim, labels_cell, _ = prepare_data(data_R, stim_level, str_side='R', label_type='med',
                                                                bool_use_dynamics=True, n_dynamics=n_dynamics)
    print('')

    # now perform the repetitions for the current dynamics length
    for idx_rep in tqdm.trange(5, leave=False, desc='SFS REP', bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:15}{r_bar}"):
        print('\n')

        # perform the sequential forward selection
        output_fin, output_init, iter_used, orig_metric = seq_forward_selection(features, y_class, y_stim, labels_cell,
                                                                        str_model='LDA', random_seed=None,
                                                                        bool_force_sfs_acc=False)

        # append to outer list
        output_med_level['sinPB'].append(output_init)
        output_med_level['sfsPB'].append(output_fin)
        print('\nHighest SinPB auc: {:.4f}'.format(output_init['vec_auc'][0]))
        print('Highest SFS auc: {:.4f}'.format(output_fin['vec_auc'][-1]))
        print('Done with rep {}'.format(idx_rep + 1))

    # append to outer list
    # initialize the dictionary if not already done
    if 'n_dynmamics_{}'.format(n_dynamics) not in output_full_level['sinPB'].keys():
        output_full_level['sinPB']['n_dynmamics_{}'.format(n_dynamics)] = []
        output_full_level['sfsPB']['n_dynmamics_{}'.format(n_dynamics)] = []
    output_full_level['sinPB']['n_dynmamics_{}'.format(n_dynamics)].append(output_med_level['sinPB'])
    output_full_level['sfsPB']['n_dynmamics_{}'.format(n_dynamics)].append(output_med_level['sfsPB'])

# shutdown ray
ray.shutdown()

# save the output
with open(str(p_output / 'RCS02_R_med_level_auc_LDA_dynamics.pkl'), 'wb') as f:
    pickle.dump(output_full_level, f)

print('debug')
