import os, hydra, yaml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import wandb
from loguru import logger
import numpy as np
import polars as pl
from collections import OrderedDict

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

# Libraries for (cross) validation
import sklearn.model_selection as skms

# Libraries for hyperparameter tuning
import optuna

# Global Variables
POTENTIAL_FEATURE_LIBRARIES = [np, tdp, tdf, skpp, skd, skfs, imos, imus, scisig, scistats]
VALIDATION_LIBRARIES = [skms]


def setup(cfg: DictConfig):
    # Pseudo-code
    # 1. Parse config file
    wandb.login()
    config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    logger.add(
        config['log_file'],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
        level="INFO",
    )
    
    # 2. Log config file to wandb, set up hydra logging, and save to disk
    if not config['hyperparam_args']['run_search'] and config['hyperparam_args']['run_search'] != 'Optuna':
        wandb.config = config
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, dir=config['run_dir'])
        
        # TODO: How to log all this during hyperparameter search?
        logger.info(f"Beginning pipeline...")
        logger.info('Local Directory Path: {}'.format(config['run_dir']))
        logger.info('WandB run url: {}'.format(run.url))
        logger.info('WandB project: {}'.format(run.project))
        logger.info('WandB entity: {}'.format(run.entity))
        logger.info('WandB run name: {}'.format(run.name))
        logger.info('WandB run id: {}'.format(run.id))
    
    return config, logger


@hydra.main(version_base=None, config_path="../conf/pipeline", config_name="pipeline")
def main(cfg: DictConfig):
    # Set up WandB and logging
    config, logger = setup(cfg)
    
    # 3. Load data (save data versioning somewhere)
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = load_data(config['data_source'])
    
    
    # 4. Preprocess data
    # The benefit of polars for preprocessing is labels will be preserved with arbitrary windowing operations on time series data
    preproc = config['preprocessing_steps']
    preproc_pipe = zip([convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func) for func in preproc.keys()], 
                    [(values) for values in preproc.values()])
    for pipe_step in preproc_pipe:
        logger.info(f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}")
        data_df = data_df.pipe(pipe_step[0], **pipe_step[1])
        
    channel_options = config['channel_options']
    # Convert to numpy
    if channel_options['stack_channels']:
        logger.info(f"Stacking Channels")
        X = np.stack([data_df.get_column(col).to_numpy() for col in config['feature_columns']], axis=1)
    else:
        X = np.concatenate([data_df.get_column(col).to_numpy() for col in config['feature_columns']], axis=1)

    # Manage labels
    if config['label_remapping']:
        logger.info(f"Remapping labels with {config['label_remapping']}")
        data_df = data_df.with_columns(pl.col(config['label_column']).map_dict(config['label_remapping']))
    y = data_df.get_column(config['label_column']).to_numpy().squeeze()
    
    # Extract group labels for LeaveOneGroupOut cross validation (if desired)
    if config['group_column']:
        logger.info(f"Using group labels for cross validation")
        groups = data_df.get_column(config['group_column']).to_numpy().squeeze()
        logger.info(f"Unique groups: {np.unique(groups)}")
    else:
        groups = None
        
    # 5. Develop feature engineering pipeline
    # TODO: Class imbalance stuff (e.g. SMOTE) would go here (probably, but how to pass y into pipe)...
    feature_steps = config['feature_engineering_steps']
    function_calls = tuple(zip([convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func) for func in feature_steps.keys()], 
                            [(values) for values in feature_steps.values()]))
    feature_steps = OrderedDict(zip(feature_steps.keys(), function_calls))
    
    feature_pipe, pipe_step_names = create_transform_pipeline(feature_steps)
    pipe_string = '\n'.join([f"{step}" for step in pipe_step_names])
    
    # Apply feature extraction pipeline on data
    logger.info(f"Transforming data into features with pipeline: \n {pipe_string}")
    if channel_options['pipe_by_channel'] & channel_options['concat_channel_features']:
        X = np.concatenate([feature_pipe.fit_transform(X[:,i] for i in range(X.shape[1]))], axis=1)
    elif channel_options['pipe_by_channel']:
        X = np.stack([feature_pipe.fit_transform(X[:,i] for i in range(X.shape[1]))], axis=1)
    else:
        X = feature_pipe.fit_transform(X)
    
    logger.info(f"Feature matrix shape after feature engineering: {X.shape}")
    logger.info(f"Label vector shape: {y.shape}")
    
    # 5a: Select Features using Boruta, SFS, etc...
    
    
    # Save and visualize feature distributions
    
    # Train / test split (K-fold cross validation, Stratified K-fold cross validation, Group K-fold cross validation)
    # TODO: Set up train / test split and vanilla hold-out validation
    # Set up cross validation object
    cv = set_up_cross_validation(config['model_validation']['method'], VALIDATION_LIBRARIES)
    if isinstance(cv, skms.LeaveOneGroupOut):
        cv = cv.split(X, y, groups)
    

    # 6. Select model
    # Note: Can use ArbitraryModel class to wrap any model and compare in pipeline with other models
    model_kwargs = config['model_kwargs'] if config['model_kwargs'] else {}
    model_args = (X, y, cv, config['model_validation']['scoring'], model_kwargs)
    model_class = find_and_load_class('model', config['model'], model_args)
    
    
    # 7. Train and evaluate model (log to wandb)
        # How to pick model? Should this be in config file? (probably)
        # 6a. Use WandB sweep to tune hyperparameters OR Use Optuna
        # WandB: `sweep_config` can be created as yaml or dict and passed to `wandb.sweep()`)
        # (Optuna integration with wandb loggin: use callback https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.WeightsAndBiasesCallback.html
                # https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
    hyperparam_args = config['hyperparam_args']
    if hyperparam_args['run_search'] and hyperparam_args['hyperparam_search_method'] == 'wandb':
        sweep_config = config['sweep_config']
        sweep_id = wandb.sweep(sweep_config, project=config['wandb']['project'], entity=config['wandb']['entity'])
        logger.info('Local Directory Path: {}'.format(config['run_dir']))
        logger.info('WandB project: {}'.format(cfg.wandb.project))
        logger.info('WandB entity: {}'.format(cfg.wandb.entity))
        logger.info(f'WandB sweep id: {sweep_id}')
        logger.info(f'WandB sweep url: https://wandb.ai/{cfg.wandb.entity}/{cfg.wandb.project}/sweeps/{sweep_id}')
        logger.info(f'WandB sweep config: {sweep_config}')
        model_class.set_output_dir(config['run_dir'])
        if sweep_config['method'] == 'grid':
            wandb.agent(sweep_id, function=model_class.wandb_train)
        else:
            wandb.agent(sweep_id, function=model_class.wandb_train, count=hyperparam_args['num_runs'])
        
    elif hyperparam_args['run_search'] and hyperparam_args['hyperparam_search_method'] == 'Optuna':
        study = optuna.create_study(direction='maximize')
        # This gets complex, how to pass in model, data, cross-validation folding, etc...?
        objective = (model_class.optuna_objective(), config)
        study.optimize(objective, n_trials=100)
    
    else:
        model_class.train()
        # TODO: Emmulate following from https://docs.wandb.ai/guides/integrations/lightgbm, especially call-back functionality
        # from wandb.lightgbm import wandb_callback, log_summary
        # import lightgbm as lgb

        # # Log metrics to W&B
        # gbm = lgb.train(..., callbacks=[wandb_callback()])

        # # Log feature importance plot and upload model checkpoint to W&B
        # log_summary(gbm, save_model_checkpoint=True)
        
    # 8. Save model (log to wandb)
    

if __name__ == "__main__":
    main()