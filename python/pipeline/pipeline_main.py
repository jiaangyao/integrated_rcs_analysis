# Full Pipeline for running unconstrained (i.e. not restricted to RC+S device capabilities) classification pipelines
# TODO: Potentially helpful to use Kedro for pipeline management

# Add python directory to path
import sys, os, time

sys.path.append(os.getcwd())

# External Imports
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from collections import OrderedDict

# Local Imports
from utils.pipeline_utils import *
from pipeline.sub_pipelines import (
    logging_setup,
    biomarker_pipeline,
    feature_engineering_pipeline,
    outlier_rejection_pipeline,
    class_imbalance_pipeline,
    data_augmentation_pipeline,
    io_pipeline,
    preproc_pipeline,
    hyperparam_opt_pipeline,
    test_model_pipeline,
)
from utils.save_data import save_data_check
from dataset.data_class import MLData
from utils.file_utils import add_config_to_csv

# Libraries for model evaluation
from training_eval.model_evaluation import (
    create_eval_class_from_config,
    implement_train_test_split,
)

# Libraries for model selection
#from model.torch_model.skorch_model import SkorchModel

# Libraries for hyperparameter tuning

# Variables
#DEFAULT_CONFIG_PATH = "../conf_example"
DEFAULT_CONFIG_PATH = "/home/claysmyth/code/configs/consolidated/"
DEFAULT_CONFIG_NAME = "pipeline_main"


# Main pipeline function
@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: DictConfig):
    # Convert config to ordered dict
    config = OrderedDict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # Setup logging, WandB, and save code, git info, and config file to run directory
    logger = setup_logging(config)

    # Load data (save data versioning somewhere)
    logger.info(f"Loading data with data params {config['data_source']}")
    data_df = io_pipeline.load_data(config["data_source"])

    # Preprocess data
    if preproc_config := config.get("preprocessing"):
        data_df = preproc_pipeline.preprocess_dataframe(
            data_df, preproc_config["functions"], logger
        )
        if feature_extraction_config := preproc_config.get(
            "feature_extraction_options"
        ):
            channel_options, feature_columns, label_options = (
                feature_extraction_config["channel_options"],
                feature_extraction_config["feature_columns"],
                preproc_config["label_options"],
            )
            X, y, one_hot_encoded, groups = preproc_pipeline.get_features_and_labels(
                data_df, channel_options, feature_columns, label_options, logger
            )
        else:
            logger.info(
                "No feature extraction from dataframe config used. Setting X (features) and y (labels) to None"
            )
            X, y, one_hot_encoded, groups = None, None, False, None

    # Feature Engineering
    if feature_eng_config := config.get("feature_engineering"):
        (
            feature_pipe,
            pipe_string,
        ) = feature_engineering_pipeline.create_feature_pipe_object(
            feature_eng_config["functions"]
        )
        num_channels, num_rows = len(feature_columns), data_df.height
        stacked_channels = channel_options["stack_channels"]
        X = feature_engineering_pipeline.process_features(
            X,
            feature_pipe,
            pipe_string,
            feature_eng_config["channel_options"],
            stacked_channels,
            num_channels,
            num_rows,
            logger,
        )
        logger.info(f"Feature matrix shape after feature engineering: {X.shape}")
        logger.info(f"Label vector shape: {y.shape}")
    
    # Reject outliers or artifacts. Note, this is a pipeline, so it is possible to run multiple outlier rejection functions in sequence.
    # The resulting dataset will those data points that pass EVERY outlier rejection function.
    if rejection_config := config.get("outlier_artifact_rejection"):
        X, y, groups = outlier_rejection_pipeline.reject_outliers_artifacts(
            X,
            y,
            groups,
            rejection_config["functions"],
            logger,
        )
        logger.info(f"Feature matrix shape after outlier and/or artifact rejection: {X.shape}")
        logger.info(f"Label vector shape: {y.shape}")


    # Set up data object once all preprocessing and feature engineering is complete
    data = MLData(X=X, y=y, groups=groups, one_hot_encoded=one_hot_encoded)

    # Implement train-test split, if desired. Comes before class imbalance correction, because class imbalance correction should only be run on training data.
    if evaluation_config := config.get("evaluation"):
        implement_train_test_split(evaluation_config, data, logger)
    
    # Evaluation Setup (i.e. CV and training folds, scoring metrics, etc...)
    if evaluation_config := config.get("evaluation"):
        eval = create_eval_class_from_config(evaluation_config, data)

    # Data Augmentation and/or Class Imbalance Correction
    augment_config = config.get("data_augmentation")
    imb_config = config.get("class_imbalance")
    if (augment_config is not None) or (imb_config is not None):
        new_folds = []
        groups_folds = []
        logger.info(f"Running data augmentation and/or class imbalance correction on each fold of training data individually. {len(data.folds)} folds in total.")
        for i in range(len(data.folds)):
            # ! No Concept of fold for groups yet! Remove entirely? Don't really need groups for class imb and data aug... 
            X_train, y_train, X_val, y_val = data.get_fold(i)
            groups_train, groups_val = data.get_groups_fold(i)
            
            if i == 0:
                logger.info(f"Original training fold shape: {X_train.shape}")
                
            # Data Augmentation
            if augment_config:
                (
                    X_train,
                    y_train,
                    groups_train,
                ) = data_augmentation_pipeline.run_data_augmentation( 
                    X_train, y_train, groups_train, augment_config, logger # ! Remove need to pass in groups... that is deprecated
                )
                
                if i == 0:
                    logger.info(f"Training fold shape after data augmentation:  {X_train.shape}")
            
            # Class Imbalance Correction
            if imb_config:
                (
                    X_train,
                    y_train,
                    groups_train,
                ) = class_imbalance_pipeline.run_class_imbalance_correction(
                    X_train, y_train, groups_train, imb_config, logger
                )
                
                if i == 0:
                    logger.info(f"Training fold shape after class imbalance correction: {X_train.shape}")
            
            new_folds.append((X_train, y_train, X_val, y_val))
            groups_folds.append((groups_train, groups_val))
        
        logger.info("Data augmentation and/or class imbalance correction complete... overriding folds with folds containing corrected training data.")
        # NOTE THAT data.X_train and data.y_train are not updated/changed here! This only updates the folds...
        data.override_folds(new_folds)
        data.override_groups_folds(groups_folds)
        logger.info(f"New training fold shape: {data.get_fold(0)[0].shape}")
    
    # Feature Selection
    # Not implemented yet

    # # Evaluation Setup (i.e. CV and training folds, scoring metrics, etc...)
    # if evaluation_config := config.get("evaluation"):
    #     eval = create_eval_class_from_config(evaluation_config, data)

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
        # TODO: SkorchModel is not fully implemented yet
        if evaluation_config and evaluation_config["model_type"] == "skorch":
            raise NotImplementedError("SkorchModel is not fully implemented yet.")
            # model_class = SkorchModel(model_class)
        if model_class is None:
            raise ValueError(f"Model class {model_name} was not found. This is typically due to an import error in a model module script...")

    # Hyperparameter Tuning and/or Model Training
    # Note: If hyperparameter_optimization field 'search_library' is not specified,
    # then the model will be trained and evaluated with default hyperparameters defined in model yaml file
    if config.get("hyperparameter_optimization") is not None:
        (
            sweep_url,
            sweep_id,
            best_run_config,
        ) = hyperparam_opt_pipeline.run_hyperparameter_search(
            config, model_class, data, eval, logger
        )

    # Test model on test (i.e. hold-out) set
    if test_model_config := config.get("test_model"):

        model_instantiation = test_model_config.get("model_instantiation")

        # Pass best_run_config to test_model_config, if desired
        if model_instantiation == "from_WandB_sweep" and 'best_run_config' in locals():
            test_model_config["best_run_config"] = best_run_config
        else:
            best_run_config = None

        # Re-initialze setup and add extra 'test' tag and 'test' info to indicate that this is a test run
        if test_model_config.get("reinit_wandb"):

            # Close previous WandB
            try:
                # wandb.teardown()
                wandb.finish()
                print("Tearing down WandB sweep and closing WandB.")
            except Exception as e:
                logger.error(f"Error occurred while tearing down WandB: {e}... note this may cause issues with reinitializing WandB during test model run. True model params may not reflect in W&B dashboard.")
                logger.error(f"Attempting to call wandb.finish()")
                try:
                    wandb.finish()
                except Exception as e:
                    logger.error(f"Error occurred while closing WandB: {e}... ignoring and continuing.")

            if "test" not in config["setup"]["wandb"]["tags"]:
                config["setup"]["wandb"]["tags"].append("test")

            # Reinitialize WandB
            # if sweep_id:
            #     wandb_test_config = config["setup"]["wandb"].copy()
            #     wandb_test_config["group"] = wandb_test_config["group"] + "_test"
            logging_setup.wandb_setup(config["setup"], config["setup"]["wandb"])
            if best_run_config:
                wandb.config.update(best_run_config)

        # Train model on entire training set, then test model on test (i.e. hold-out) set and log results.
        # Note, that data augmentation where run on each fold individually, to avoid data leakage into the validation sets of each fold.
        # Now, we are training the model on the entire training set, so we need to run data augmentation on the entire training set, then test on the test set.
        if (augment_config is not None) or (imb_config is not None):
            X_train, y_train = data.get_training_data()
            logger.info(f"Original training set shape: {X_train.shape}")
            
            # Data Augmentation
            if augment_config:
                (
                    X_train,
                    y_train,
                    _
                ) = data_augmentation_pipeline.run_data_augmentation( 
                    X_train, y_train, data.groups_train, augment_config, logger # ! Remove need to pass in groups... that is deprecated
                )
                
                logger.info(f"Training set shape after data augmentation:  {X_train.shape}")
            
            # Class Imbalance Correction
            if imb_config:
                (
                    X_train,
                    y_train,
                    _,
                ) = class_imbalance_pipeline.run_class_imbalance_correction(
                    X_train, y_train, data.groups_train, imb_config, logger
                )
                
                logger.info(f"Training set shape after class imbalance correction: {X_train.shape}")
            
            data.X_train, data.y_train = X_train, y_train
            
        # Train model on entire training set, then test model on test (i.e. hold-out) set and log results.
        test_model_pipeline.test_model(
            model_class, eval, data, config, test_model_config, logger
        )

        # Log entry to pipeline runs tracking csv

    # Save Model
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

    try:
        # First add log file to WandB
        log_artifact = wandb.Artifact("local_logs", type="log")
        log_artifact.add_file(local_path=config['setup'].get("file_log"))
        log_artifact.save(log_artifact)
        # Close WandB
        wandb.finish()
    except Exception as e:
        logger.error(f"Error occurred while closing WandB: {e}... ignoring and continuing.")


if __name__ == "__main__":
    # # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config-path", help="Path to config directory")
    # args, _ = parser.parse_known_args()

    # # Set DEFAULT_CONFIG_PATH if --config-path flag is provided
    # if args.config_path:
    #     DEFAULT_CONFIG_PATH = args.config_path
    # try:
    #     main()
    # except wandb.sdk.wandb_manager.ManagerConnectionRefusedError as e:
    #     print("Ran into a WandB error... Sleeping for 3 min and trying again.")
    #     time.sleep(180)
    #     main()
    
    main()
        
    
