import pandas as pd
import polars as pl
import pickle
import importlib
import os
from utils.file_utils import ensure_parent_directory_exists
from polars import selectors as cs

def save_data(X, config_dict):
    """
    Saves data according to the configuration provided in the YAML files.

    :param X: Input data, can be a Polars DataFrame, Pandas DataFrame, or an instance of a class.
    :param config_dict: Dictionary containing configuration for saving data.
    :param subbase_path: Base path for file saving.
    :param device: Device identifier used in file naming.
    :return: None
    """
    # Check input type and output type
    input_type = config_dict.get('input_type', '')
    output_type = config_dict.get('output_type', '')

    file_path = config_dict.get('file_path', '')
    # Save data according to the output type
    if file_path != '':
        ensure_parent_directory_exists(file_path)
        if output_type == 'parquet' and isinstance(X, pd.DataFrame):
            X.to_parquet(file_path)
        elif output_type == 'parquet' and isinstance(X, pl.DataFrame):
            X.write_parquet(file_path, use_pyarrow=True)
        elif output_type == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(X, f)
    elif output_type == 'database':
            module = importlib.import_module(config_dict['database_module'])
            con = module.connect(config_dict['database_path'])
            if config_dict['to_arrow']:
                X = X.to_arrow()
            con.sql(config_dict['query'])
            con.close()
    else:
        raise ValueError("Unsupported data type or output format, or file path not specified.")

def save_data_check(df, data_class, config_dict, logger=None):
    
    # If config_dict is None, then user specified to not save data
    if config_dict is None:
        return
    
    if logger: logger.info(f"Saving data with params... {config_dict}")
    # Check input type
    input_type = config_dict.get('input_type', '')
    input_type = input_type.lower()
    if input_type == 'dataframe' or input_type == 'df' or input_type == 'data_frame':
        save_data(df, config_dict)
    elif input_type == 'dataclass' or input_type == 'data_class':
        save_data(data_class, config_dict)
    else:
        raise ValueError(f"Unsupported input type: {input_type}.")
        