# Full Pipeline for running unconstrained (i.e. not restricted to RC+S device capabilities) classification pipelines
# TODO: Potentially helpful to use Kedro for pipeline management

# External Imports
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Local Imports
from utils.pipeline_utils import *
from sub_pipelines import (
    logging_setup,
    biomarker_pipeline,
    feature_engineering_pipeline,
    io_pipeline,
    preproc_pipeline,
    hyperparam_opt_pipeline,
)
from utils.save_data import save_data_check
from dataset.data_class import MLData
from utils.file_utils import add_config_to_csv

# Libraries for model evaluation
from training_eval.model_evaluation import create_eval_class_from_config

# Libraries for model selection
from model.torch_model.skorch_model import SkorchModel

# Libraries for hyperparameter tuning
from training_eval.hyperparameter_optimization import HyperparameterOptimization

# Variables
CONFIG_PATH = "/home/claysmyth/code/configs/lightgbm_sleep"
CONFIG_NAME = "pipeline_main"


# Main pipeline function
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME,
)
def main(cfg: DictConfig):
    # Logging Setup
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Check if hyperparameter optimization is desired... needs to be passed to setup
    if hyperopt_conf := config.get("hyperparameter_optimization"):
        if hyperopt_conf.get("search_library").lower() == "wandb":
            WandB_hyperopt = True
    else:
        WandB_hyperopt = False
    logger = logging_setup.setup(config["setup"], WandB_hyperopt=WandB_hyperopt)

    # Load data (save data versioning somewhere)
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = io_pipeline.load_data(config["data_source"])

    # Preprocess data
    preproc_config = config["preprocessing"]
    data_df = preproc_pipeline.preprocess_dataframe(
        data_df, preproc_config["functions"], logger
    )
    channel_options, feature_columns, label_options = (
        preproc_config["feature_extraction_options"]["channel_options"],
        preproc_config["feature_extraction_options"]["feature_columns"],
        preproc_config["label_options"],
    )
    X, y, one_hot_encoded, groups = preproc_pipeline.get_features_and_labels(
        data_df, channel_options, feature_columns, label_options, logger
    )

    # Feature Engineering
    feature_eng_config = config["feature_engineering"]
    feature_pipe, pipe_string = feature_engineering_pipeline.create_feature_pipe_object(
        feature_eng_config["functions"]
    )
    num_channels, num_rows = len(feature_columns), data_df.height
    stacked_channels = channel_options["stack_channels"]
    X = feature_engineering_pipeline.process_features(
        X, feature_pipe, pipe_string, feature_eng_config['channel_options'], stacked_channels, num_channels, num_rows, logger
    )
    logger.info(f"Feature matrix shape after feature engineering: {X.shape}")
    logger.info(f"Label vector shape: {y.shape}")

    # Feature Selection
        # Not implemented yet

    # Class Imbalance
        # Not implemented yet

    # Set up data object once all preprocessing and feature engineering is complete
    data = MLData(X=X, y=y, groups=groups, one_hot_encoded=one_hot_encoded)
    # Save data, if desired
    save_data_check(data_df, data, config.get("save_dataframe_or_features"), logger)
    
    # Visualize data, if desired
        # Not implemented yet

    # Evaluation Setup (i.e. CV, scoring metrics, etc...)
    evaluation_config = config["evaluation"]
    eval = create_eval_class_from_config(evaluation_config, data)

    # Model Setup
    model_config = config["model"]
    model_name = model_config["model_name"]
    model_kwargs = model_config["parameters"] if model_config["parameters"] else {}
    if early_stopping := model_config.get("early_stopping"):
        model_kwargs["early_stopping"] = early_stopping

    model_class = find_and_load_class("model", model_name, kwargs=model_kwargs)
    if evaluation_config["model_type"] == "skorch":
        model_class = SkorchModel(model_class)
    

    # Hyperparameter Tuning and/or Model Training
    # Note: If hyperparameter_optimization is not specified, 
    # then the model will be trained and evaluated with default hyperparameters defined in model yaml file
    sweep_url, sweep_id = hyperparam_opt_pipeline.run_hyperparameter_search(
        config, model_class, data, eval, logger
    )
    
    # Test model on test set
        # Not implemented yet

    # Add run to pipeline runs tracking csv
    if config.get("run_tracking_csv") is not None:
        
        # Check if sweep_url and sweep_id are defined
        if "sweep_url" in locals() and sweep_url is not None:
            wandb_url = sweep_url
        elif wandb.run.url:
            wandb_url = wandb.run.url
        else:
            wandb_url = None
        
        if "sweep_id" in locals() and sweep_id is not None:
            wandb_id = sweep_id
        elif wandb.run.id:
            wandb_id = wandb.run.id
        else:
            wandb_id = None

        # Add config to tracking csv
        add_config_to_csv(
            config | {"WandB_url": wandb_url, "WandB_id": wandb_id},
            config["run_tracking_csv"],
        )

    # Close WandB
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
