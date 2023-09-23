import os, hydra, yaml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import wandb
from loguru import logger
import numpy as np
import polars as pl

# Stand-in variable for custom time domain processing functions
import preproc.time_domain_processing as tdp
import preproc.time_domain_features as tdf

# Libraries for preprocessing and feature selection
import sklearn.preprocessing as skpp
import sklearn.decomposition as skd
import sklearn.feature_selection as skfs
import imblearn.over_sampling as imos
import imblearn.under_sampling as imus
import scipy.signal as scisig
import scipy.stats as scistats
from utils.pipeline_utils import *

# Libraries for hyperparameter tuning
import optuna

# Global Variables
POTENTIAL_FEATURE_LIBRARIES = [tdp, tdf, skpp, skd, skfs, imos, imus, scisig, scistats]


def setup(cfg: DictConfig):
    # Pseudo-code
    # 1. Parse config file
    wandb.login()
    config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    # 2. Log config file to wandb, set up hydra logging, and save to disk
    wandb.config = config
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, dir=config['run_dir'])

    logger.add(
        config['log_file'],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
        level="INFO",
    )
    
    logger.info(f"Beginning pipeline...")
    logger.info('WandB run url: {}'.format(run.url))
    logger.info('WandB project: {}'.format(run.project))
    logger.info('WandB entity: {}'.format(run.entity))
    logger.info('WandB run name: {}'.format(run.name))
    logger.info('WandB run id: {}'.format(run.id))
    
    return config, logger


@hydra.main(version_base=None, config_path="../conf/pipeline", config_name="clay_debug")
def main(cfg: DictConfig):
    config, logger = setup(cfg)
    
    # Log the overall steps chosen (dataset, preprocessing, feature extraction, feature selection, dealing with class imbalance,
    # train / test split, model, hyperparameter sweeping)
    
    # 3. Load data (save data versioning somewhere)
        # 3a. Use UI to select data?
        # Convert to polars df
    
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = load_data(config['data_source'])
    
    
    # 4. Preprocess data (log this somewhere too, hydra+loguru logging and/or wandb logging) (conver to numpy??)
    # if use_polars:
    
    # The benefit of polars is that if you decide to aggregate over windows, then it will automatically keep labels with the data
    preproc = config['preprocessing_steps']
    preproc_pipe = zip([convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func) for func in preproc.keys()], 
                    [(values) for values in preproc.values()])
    for pipe_step in preproc_pipe:
        logger.info(f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}")
        data_df = data_df.pipe(pipe_step[0], **pipe_step[1])
        
    # else:
    #     preproc = config['preprocessing_steps']
    #     preproc_pipe = zip((convert_string_to_callable(tdp, func) for func in preproc.keys()), ((values) for values in preproc.values()))
    #     for preproc_func, preproc_args in preproc_pipe:
    #         X = np.preproc_func(X, **preproc_args)
    
    # Convert to numpy
    X = data_df.select(pl.all().exclude(config['label_column'])).to_numpy()
    if config['label_remapping']:
        logger.info(f"Remapping labels with {config['label_remapping']}")
        data_df = data_df.with_columns(pl.col(config['label_column']).map_dict(config['label_remapping']))
    y = data_df['label_column'].to_numpy().squeeze()
    
    # 5. Extract features (E.g. convert to spectral domain)
    # Class imbalance stuff (e.g. smote) would go here (probably)...
    feature_steps = config['feature_engineering_steps']
    feature_steps = dict(tuple(zip[feature_steps.keys(), 
                            tuple(zip([convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func) for func in feature_steps.keys()], 
                            [(values) for values in feature_steps.values()]))
                        ]))
    
    feature_pipe = create_transform_pipeline(feature_steps)
    pipe_steps = '\n'.join([f"{name}: {transformer}" for name, transformer in feature_pipe.steps])
    logger.info(f"Transforming data into features with pipeline {pipe_steps}")
    # Apply feature extraction pipeline on data
    X = feature_pipe.fit_transform(X)
    
    # Select Features from Boruta, SFS, etc...
    
    # Save and visualize feature distributions
    
    # Train / test split (K-fold cross validation, Stratified K-fold cross validation, Group K-fold cross validation)
    # TODO: Incorporate this into model class so that it can be used for hyperparameter tuning (e.g. logged into WandB)

    # Select model
    # might need to use create_instance_from_module instead, neither functions are tested as they are LLM generated
    model_class = create_instance_from_directory(config['model'], X, y) 
    
    
    # 6. Train model (log to wandb)
        # How to pick model? Should this be in config file? (probably)
        # 6a. Use WandB sweep to tune hyperparameters OR Use Optuna
        # WandB: `sweep_config` can be created as yaml or dict and passed to `wandb.sweep()`)
        # (Optuna integration with wandb loggin: use callback https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.WeightsAndBiasesCallback.html
                # https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
    # 7. Evaluate model (log to wandb, save custom visualizations)
    if config['hyperparam_search'] == 'wandb':
        sweep_config = config['wandb']['sweep_config']
        # TODO: How to pass in hyperparameters to wandb sweep via config
        sweep_id = wandb.sweep(sweep_config, project=config['wandb']['project'], entity=config['wandb']['entity'])
        # TODO: Where to put train function?
        wandb.agent(sweep_id, function=model_class.wandb_objective)
        
    elif config['hyperparam_search'] == 'Optuna':
        study = optuna.create_study(direction='maximize')
        # This gets complex, how to pass in model, data, cross-validation folding, etc...?
        objective = (model_class.optuna_objective(), config)
        study.optimize(objective, n_trials=100)
    
    else:
        # Train with default parameters
        None
        
    # 8. Save model (log to wandb)
    

if __name__ == "__main__":
    main()