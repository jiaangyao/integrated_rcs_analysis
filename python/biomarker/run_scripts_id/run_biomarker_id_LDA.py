import pathlib
import pickle

import ray
import tqdm
import pandas as pd
from easydict import EasyDict as edict

from biomarker.training.prepare_data import prepare_data
from biomarker.training.seq_forward_selection import seq_forward_selection
from utils.parse_datetime import parse_dt_w_tz
from biomarker.run_scripts_id.biomarker_id import biomarker_id_side


def gen_config_LDA():
    """_summary_

    Returns:
        dict: Configurations for biomarker identification using different subjects
    """
    cfg = dict()


    return cfg


def tempty():
    """"""

# hardcode path to the RCS02 step 3 data for now
p_project = pathlib.Path('/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/')
p_data = p_project / 'Data/RCS02/Step3_in_clinic_neural_recordings/'
p_output = pathlib.Path('/home/jyao/Downloads/biomarker_id/model_id/')
f_data_L = 'rcs02_L_table.csv'
f_data_R = 'rcs02_R_table.csv'

str_side = 'R'
str_model = 'LDA'



