# Full Pipeline for running unconstrained (i.e. not restricted to RC+S device capabilities) classification pipelines
# TODO: Potentially helpful to use Kedro for pipeline management

# External Imports
import hydra
from omegaconf import DictConfig
import wandb

# Local Imports
from utils.pipeline_utils import *
from sub_pipelines import logging_setup, biomarker_pipeline, feature_engineering_pipeline, io_pipeline, preproc_pipeline, hyperparam_opt_pipeline
from utils.save_data import save_data_check
from dataset.data_class import MLData
from utils.file_utils import add_config_to_csv

# Libraries for model evaluation
from training_eval.model_evaluation import create_eval_class_from_config

# Libraries for model selection
from model.torch_model.skorch_model import SkorchModel

# Libraries for hyperparameter tuning
from training_eval.hyperparameter_optimization import HyperparameterOptimization

# Logging Imports
from analysis_CS.process_classification_results import process_and_log_eval_results_sklearn, process_and_log_eval_results_torch

# Main pipeline function
@hydra.main(
    version_base=None,
    config_path="../REPLACE_ME",
    config_name="REPLACE_ME",
)
def main(cfg: DictConfig):
    # Logging Setup
    config, logger = logging_setup.setup(cfg)

    # 3. Load data (save data versioning somewhere)
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = io_pipeline.load_data(config["data_source"])
    
    # Preprocess data
    preproc_config = config["preprocessing"]
    data_df = preproc_pipeline.preprocess_dataframe(data_df, preproc_config['functions'], logger)
    channel_options, feature_columns, label_options = preproc_config["channel_options"], preproc_config['feature_columns'], preproc_config['label_options']
    X, y, one_hot_encoded, groups = preproc_pipeline.get_features_and_labels(data_df, channel_options, feature_columns, label_options, logger)
    
    # Feature Engineering
    feature_eng_config = config["feature_engineering"]
    feature_pipe, pipe_string = feature_engineering_pipeline.create_feature_pipe_object(feature_eng_config["functions"], logger)
    num_channels, num_rows = len(feature_columns), data_df.height
    X = feature_engineering_pipeline.process_features(X, feature_pipe, pipe_string, channel_options, num_channels, num_rows, logger)
    logger.info(f"Feature matrix shape after feature engineering: {X.shape}")
    logger.info(f"Label vector shape: {y.shape}")
    
    # Feature Selection

    # Class Imbalance
    
    # Set up data object once all preprocessing and feature engineering is complete
    data = MLData(X=X, y=y, groups=groups, one_hot_encoded=one_hot_encoded)
    # Save data, if desired
    save_data_check(data_df, data, config.get("save_dataframe_or_features"), logger)

    # Evaluation Setup (i.e. CV, metrics, etc...)
    evaluation_config = config["evaluation"]
    eval = create_eval_class_from_config(evaluation_config, data)

    # Model Setup
    model_config = config["model"]
    model_name = model_config["model_name"]
    model_kwargs = model_config["parameters"] if model_config["parameters"] else {}
    if early_stopping := model_config.get("early_stopping"): model_kwargs["early_stopping"] = early_stopping
    
    model_class = find_and_load_class("model", model_name, kwargs=model_kwargs)
    if evaluation_config["model_type"] == "skorch":
        model_class = SkorchModel(model_class)

    # Training or Hyperparameter Tuning
    hyperparam_config = config["hyperparameter_optimization"]
    if hyperparam_config["run_search"]:
        # Hyperparameter search. First, check which library to use
        ho = HyperparameterOptimization(model_class, data, eval)
        
        if hyperparam_config["search_library"].lower() == "wandb":
            sweep_id, sweep_config, sweep_url = hyperparam_opt_pipeline.wandb_sweep_setup(eval, ho, data, config, logger)
            hyperparam_opt_pipeline.run_wandb_sweep(ho, sweep_config["method"], hyperparam_config["num_runs"], sweep_id)
            
        elif hyperparam_config["search_library"].lower() == "optuna":
            raise NotImplementedError
        
    else:
        # Train model and Evaluate predictions
        results, epoch_metrics = eval.evaluate_model(
            model_class, data
        )
        
        # Log results (to WandB)
        # TODO: Create a logging class to handle this, so that users can pick different logging options/dashboards outside of W&B
        if eval.model_type == 'torch':
            process_and_log_eval_results_torch(results, config["run_dir"], epoch_metrics)
        elif eval.model_type == 'sklearn':
            process_and_log_eval_results_sklearn(results, config["run_dir"])
    
    
    # Add run to pipeline tracking csv
    if config.get("run_tracking_csv") is not None:

        if 'sweep_url' in locals():
            wandb_url = sweep_url
        elif wandb.run.url:
            wandb_url = wandb.run.url
            
        add_config_to_csv(config | {"WandB_url": wandb_url, "WandB_id": sweep_id}, config["run_tracking_csv"])
    
    # Close WandB
    if wandb.run: 
        wandb.finish()


if __name__ == "__main__":
    main()
    
    