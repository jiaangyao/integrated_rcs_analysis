# Full Pipeline for running unconstrained (i.e. not restricted to RC+S device capabilities) classification pipelines
# TODO: Potentially helpful to use Kedro for pipeline management

# Add python directory to path
import sys,os
sys.path.append(os.getcwd())

# External Imports
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from collections import OrderedDict

# Local Imports
from utils.pipeline_utils import *
from sub_pipelines import (
    logging_setup,
    biomarker_pipeline,
    feature_engineering_pipeline,
    class_imbalance_pipeline,
    io_pipeline,
    preproc_pipeline,
    hyperparam_opt_pipeline,
    test_model_pipeline,
)
from utils.save_data import save_data_check
from dataset.data_class import MLData
from utils.file_utils import add_config_to_csv

# Libraries for model evaluation
from training_eval.model_evaluation import create_eval_class_from_config

# Libraries for model selection
from model.torch_model.skorch_model import SkorchModel

# Libraries for hyperparameter tuning

# Variables
CONFIG_PATH = "/home/claysmyth/code/configs/lightgbm_test_LOGO_and_data_fold_consistency"
CONFIG_NAME = "pipeline_main"


# Main pipeline function
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME,
)
def main(cfg: DictConfig):
    # Convert config to ordered dict
    config = OrderedDict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    # Logging Setup
    # Check if hyperparameter optimization is desired... needs to be passed to setup
    if hyperopt_conf := config.get("hyperparameter_optimization"):
        if hyperopt_conf.get("search_library") and hyperopt_conf.get("search_library").lower() == "wandb":
            WandB_hyperopt = True
        else:
            WandB_hyperopt = False
    else:
        WandB_hyperopt = False
    logger = logging_setup.setup(config["setup"], WandB_hyperopt=WandB_hyperopt)

    # Load data (save data versioning somewhere)
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = io_pipeline.load_data(config["data_source"])

    # Preprocess data
    if preproc_config := config.get("preprocessing"):
        data_df = preproc_pipeline.preprocess_dataframe(
            data_df, preproc_config["functions"], logger
        )
        if feature_extraction_config := preproc_config.get("feature_extraction_options"):
            channel_options, feature_columns, label_options = (
                feature_extraction_config["channel_options"],
                feature_extraction_config["feature_columns"],
                preproc_config["label_options"],
            )
            X, y, one_hot_encoded, groups = preproc_pipeline.get_features_and_labels(
                data_df, channel_options, feature_columns, label_options, logger
            )
        else:
            logger.info("No feature extraction from dataframe config used. Setting X (features) and y (labels) to None")
            X, y, one_hot_encoded, groups = None, None, False, None

    # Feature Engineering
    if feature_eng_config := config.get("feature_engineering"):
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


    # Class Imbalance Correction
    if imb_config := config.get("class_imbalance"):
        X, y, groups = class_imbalance_pipeline.run_class_imbalance_correction(X, y, groups, imb_config, logger)
        
    # Feature Selection
        # Not implemented yet

    # Set up data object once all preprocessing and feature engineering is complete
    data = MLData(X=X, y=y, groups=groups, one_hot_encoded=one_hot_encoded)
    

    # Evaluation Setup (i.e. CV, scoring metrics, etc...)
    # TODO: Train/Val/Test split does not split groups... which leads to error using LOGO CV
    if evaluation_config := config.get("evaluation"):
        eval = create_eval_class_from_config(evaluation_config, data)
        
        
    # Visualize data, if desired
        # Not implemented yet
    
    # Save data, if desired. This occurs after evaluation setup because the data class (MLData) object may be modified during evaluation setup (e.g. training folds)
    save_data_check(data_df, data, config.get("save_dataframe_or_features"), logger)

    # Model Setup
    if model_config := config.get("model"):
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
    if config.get("hyperparameter_optimization") is not None:
        sweep_url, sweep_id, best_run_config = hyperparam_opt_pipeline.run_hyperparameter_search(
            config, model_class, data, eval, logger
        )
    
    # Test model on test set
    if test_model_config := config.get("test_model") and best_run_config in locals():
        
        # Pass best_run_config to test_model_config, if desired
        if test_model_config["model_options"]['model_instantiation'] == "from_WandB_sweep":
            test_model_config['model_options']['best_run_config'] = best_run_config
            
        test_model_pipeline.test_model(model_class, eval, data, test_model_config, logger)
        # 1. Train model on entire training set
        # 2. Evaluate model on test set
        # Not implemented yet
    
    # Save Model
    # TODO: Consider moving this elsewhere
    if save_model_config := config.get("save_model"):
        save_model(save_model_config, model_class.model, logger)


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
