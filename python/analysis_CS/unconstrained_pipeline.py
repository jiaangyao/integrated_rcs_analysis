import os, hydra, yaml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import wandb
from loguru import logger
import numpy as np
import polars as pl
from collections import OrderedDict
from utils.file_utils import create_zip, get_git_info, add_config_to_csv

# TODO: Remove sys.append once I figure out why VSCode hates this working directory
import sys

sys.path.append("/home/claysmyth/code/integrated_rcs_analysis/python/")

from io_module.base import load_data

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
from dataset.data_class import MLData

# Libraries for model evaluation
from training_eval.model_evaluation import create_eval_class_from_config

# Libraries for model selection
from model.torch_model.skorch_model import SkorchModel

# Libraries for hyperparameter tuning
from training_eval.hyperparameter_optimization import HyperparameterOptimization

# Global Variables
POTENTIAL_FEATURE_LIBRARIES = [
    tdp,
    tdf,
    np,
    skpp,
    skd,
    skfs,
    imos,
    imus,
    scisig,
    scistats,
]


def setup(cfg: DictConfig):
    # Pseudo-code
    # 1. Parse config file
    wandb.login()
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    logger.add(
        config["log_file"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
        level="INFO",
    )
    logger.info("Local Directory Path: {}".format(config["run_dir"]))
    # TODO: Parameterize this...
    create_zip(f'{os.getcwd()}/python', f'{config["run_dir"]}/code.zip', exclude=['analysis_*'])
    git_info = get_git_info()
    logger.info("Git info: {}".format(git_info))
    config |= git_info
    
    logger.info(f"Beginning pipeline...")

    # 2. Log config file to wandb, set up hydra logging, and save to disk
    if (
        not config["hyperparameter_optimization"]["run_search"]
        and config["hyperparameter_optimization"]["run_search"] != "Optuna"
    ):
        wandb.config = config
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            dir=config["run_dir"],
        )
        logger.info("WandB run url: {}".format(run.url))
        logger.info("WandB project: {}".format(run.project))
        logger.info("WandB entity: {}".format(run.entity))
        logger.info("WandB run name: {}".format(run.name))
        logger.info("WandB run id: {}".format(run.id))

    return config, logger


@hydra.main(
    version_base=None,
    config_path="../conf/pipeline_prototype",
    config_name="pipeline_unconstrained",
)
def main(cfg: DictConfig):
    # TODO: Move each of preproc, feature_engineering, etc.. into their own sub-pipeline functions or scripts
    # Set up WandB and logging
    config, logger = setup(cfg)

    # 3. Load data (save data versioning somewhere)
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = load_data(config["data_source"])

    # 4. Preprocess data
    # The benefit of polars for preprocessing is labels will be preserved with arbitrary windowing operations on time series data
    preproc = config["preprocessing"]
    preproc_funcs = preproc["functions"]
    preproc_pipe = zip(
        [
            convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func)
            for func in preproc_funcs.keys()
        ],
        [(values) for values in preproc_funcs.values()],
    )
    for pipe_step in preproc_pipe:
        logger.info(
            f"Running preprocessing step {pipe_step[0]} with args {pipe_step[1]}"
        )
        data_df = data_df.pipe(pipe_step[0], **pipe_step[1])

    channel_options = preproc["channel_options"]
    # Convert to numpy
    if channel_options["stack_channels"]:
        logger.info(f"Stacking Channels")
        X = np.stack(
            [data_df.get_column(col).to_numpy() for col in preproc["feature_columns"]],
            axis=1,
        )
    else:
        X = np.concatenate(
            [data_df.get_column(col).to_numpy() for col in preproc["feature_columns"]],
            axis=1,
        )

    # Manage labels
    label_config = preproc["label_options"]
    if label_config["label_remapping"]:
        logger.info(f"Remapping labels with {label_config['label_remapping']}")
        data_df = data_df.with_columns(
            pl.col(label_config["label_column"]).map_dict(
                label_config["label_remapping"]
            )
        )
    y = data_df.get_column(label_config["label_column"]).to_numpy().squeeze()

    if label_config["OneHotEncode"]:
        # TODO: Save y before one-hot encoding for later, save both to data object
        y = skpp.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        one_hot_encoded = True
    else:
        one_hot_encoded = False

    # Extract group labels for LeaveOneGroupOut cross validation (if desired)
    if label_config["group_column"]:
        logger.info(f"Using group labels for cross validation")
        groups = data_df.get_column(label_config["group_column"]).to_numpy().squeeze()
        logger.info(f"Unique groups: {np.unique(groups)}")
    else:
        groups = None

    # 5. Develop feature engineering pipeline
    # TODO: Class imbalance, or Data Augmentation,  stuff (e.g. SMOTE) would go here (probably, but how to pass y into pipe)...
    feature_eng_config = config["feature_engineering"]
    feature_eng_funcs = feature_eng_config["functions"]
    function_calls = tuple(
        zip(
            [
                convert_string_to_callable(POTENTIAL_FEATURE_LIBRARIES, func)
                for func in feature_eng_funcs.keys()
            ],
            [(values) for values in feature_eng_funcs.values()],
        )
    )
    feature_steps = OrderedDict(zip(feature_eng_funcs.keys(), function_calls))

    feature_pipe, pipe_step_names = create_transform_pipeline(feature_steps)
    pipe_string = "\n".join([f"{step}" for step in pipe_step_names])

    # Apply feature extraction pipeline on data
    logger.info(f"Transforming data into features with pipeline: \n {pipe_string}")
    if channel_options["pipe_by_channel"] & channel_options["concat_channel_features"]:
        X = np.concatenate(
            [feature_pipe.fit_transform(X[:, i] for i in range(X.shape[1]))], axis=1
        )
    elif (
        channel_options["pipe_by_channel"] & channel_options["average_channel_features"]
    ):
        X = np.mean(
            np.stack(
                [feature_pipe.fit_transform(X[:, i] for i in range(X.shape[1]))], axis=1
            ),
            axis=1,
        )
    elif channel_options["pipe_by_channel"] & channel_options["stack_channel_features"]:
        X = np.stack(
            [feature_pipe.fit_transform(X[:, i] for i in range(X.shape[1]))], axis=1
        )
    else:
        X = feature_pipe.fit_transform(X)

    logger.info(f"Feature matrix shape after feature engineering: {X.shape}")
    logger.info(f"Label vector shape: {y.shape}")

    # 5a: Select Features using Boruta, SFS, etc...

    # Save and visualize feature distributions

    
    # Set up data object once all preprocessing and feature engineering is complete
    data = MLData(X=X, y=y, groups=groups, one_hot_encoded=one_hot_encoded)
    
    # 6. Train / test split (K-fold cross validation, Stratified K-fold cross validation, Group K-fold cross validation)
    # Set up model evaluation object
    evaluation_config = config["evaluation"]
    eval, early_stopping = create_eval_class_from_config(evaluation_config, data)

    # 7. Select model
    # Note: Can use ArbitraryModel class to wrap any model and compare in pipeline with other models
    model_config = config["model"]
    model_name = model_config["model_name"]
    model_kwargs = model_config["parameters"] if model_config["parameters"] else {}
    if early_stopping: model_kwargs["early_stopping"] = early_stopping
    
    model_class = find_and_load_class("model", model_name, kwargs=model_kwargs)
    if evaluation_config["model_type"] == "skorch":
        model_class = SkorchModel(model_class)

    # 8. Train and evaluate model (log to wandb)
    # (Optuna integration with wandb logging: use callback https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.WeightsAndBiasesCallback.html
    # https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
    hyperparam_args = config["hyperparameter_optimization"]
    ho = HyperparameterOptimization(model_class, data, eval)
    if hyperparam_args["run_search"] and hyperparam_args["name"].lower() == "wandb":
        if (
            eval.model_type == "skorch"
            or eval.model_type == "torch"
        ):
            # Add input and output shape, which depends on data
            config["sweep"]["parameters"]["n_input"] = {"value": X.shape[-1]}
            # Assumes y is one-hot encoded
            # TODO: Remove this assumption
            config["sweep"]["parameters"]["n_class"] = {"value": y.shape[-1]}

        config["sweep"]["name"] = f"{cfg.run_name}_{cfg.device}_{cfg.time_stamp}_sweep"
        # config["sweep"]["local_directory"] = config["run_dir"]
        sweep_config = config["sweep"]
        sweep_id = wandb.sweep(
            sweep_config,
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
        )

        # Log relevant info
        logger.info("WandB project: {}".format(cfg.wandb.project))
        logger.info("WandB entity: {}".format(cfg.wandb.entity))
        logger.info(f"WandB sweep id: {sweep_id}")
        wandb_url = f"https://wandb.ai/{cfg.wandb.entity}/{cfg.wandb.project}/sweeps/{sweep_id}"
        logger.info(
            f"WandB sweep url: {wandb_url}"
        )
        logger.info(f"WandB sweep config: {sweep_config}")
        ho.initialize_wandb_params(
            config["run_dir"], config["wandb"]["group"], config["wandb"]["tags"]
        )
        
        # Run sweep
        if sweep_config["method"] == "grid":
            wandb.agent(sweep_id, function=ho.wandb_sweep)
        else:
            wandb.agent(
                sweep_id,
                function=ho.wandb_sweep,
                count=hyperparam_args["num_runs"],
            )
        
        add_config_to_csv(config | {"WandB_url": wandb_url, "WandB_id": sweep_id}, config["run_tracking_csv"])
        
    elif (
        hyperparam_args["run_search"]
        and hyperparam_args["hyperparam_search_method"] == "Optuna"
    ):
        ho.opunta(hyperparam_args)

    else:
        # model_class.train()
        # TODO: Emmulate following from https://docs.wandb.ai/guides/integrations/lightgbm, especially call-back functionality
        # from wandb.lightgbm import wandb_callback, log_summary
        # import lightgbm as lgb
        
        # Evaluate predictions
        results = eval.evaluate_model(
            model_class, data
        )

        # Drop prefixes for logging
        mean_results = {
            (f'{k.split("_", 1)[-1]}_mean' if 'test_' in k else f'{k}_mean'): np.mean(v) 
            for k, v in results.items()
        }
        std_results = {
            (f'{k.split("_", 1)[-1]}_std' if 'test_' in k else f'{k}_std'): np.std(v) 
            for k, v in results.items()
        }

        # Log model performance metrics to W&B
        wandb.log(std_results)
        wandb.log(mean_results)

        add_config_to_csv(config | {"WandB_url": wandb.run.url, "WandB_id": wandb.run.id}, config["run_tracking_csv"])
        # # Log metrics to W&B
        # gbm = lgb.train(..., callbacks=[wandb_callback()])

        # # Log feature importance plot and upload model checkpoint to W&B
        # log_summary(gbm, save_model_checkpoint=True)

    # 8. Save model (log to wandb)
    wandb.finish()


if __name__ == "__main__":
    main()
