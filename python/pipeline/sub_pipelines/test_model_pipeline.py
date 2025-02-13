from training_eval.process_classification_results import (
    process_and_log_eval_results_sklearn,
    process_and_log_eval_results_torch,
)


def get_model(model_class, test_model_conf, logger):
    """
    Returns the model class, either by loading the model from a file,
    resetting the model, or creating a model from the model options.
    """
    if model_class is None:
        raise ValueError("Model class is None.")

    model_instantiation = test_model_conf.get("model_instantiation")

    if model_instantiation == "reset" or model_instantiation == "default":
        logger.info("Resetting model for testing.")
        model_class.reset_model()

    elif model_instantiation == "load":
        logger.info("Loading model.")
        raise NotImplementedError("Loading models is not implemented yet.")

    elif model_instantiation == "from_WandB_sweep":

        logger.info(f"Creating model from best hyperparameters from sweep: {test_model_conf.get('best_run_config')}")

        # Create model from best hyperparameters
        model_class.override_model(test_model_conf.get("best_run_config"))

    else:
        raise ValueError(f"Please choose a supported model instantiation option.")

    return model_class


def test_model(model_class, eval, data, config, test_model_conf, logger):
    """
    Evaluates the model on the test set, and logs the results to WandB.
    """
    # Reset model, load model, or create model class from config generated in hyperparameter search
    model_class = get_model(model_class, test_model_conf, logger)

    # Train Model on whole training set, and get results on test set
    logger.info(
        "Training model on whole training set, and getting results on test set."
    )
    results, epoch_metrics = eval.test_model(model_class, data)

    # Log results (to WandB)
    # TODO: Create a logging class to handle this, so that users can pick different logging options/dashboards outside of W&B
    if eval.model_type == "torch":
        process_and_log_eval_results_torch(
            results, config["setup"]["path_run"], epoch_metrics, test=True
        )
    elif eval.model_type == "sklearn":
        process_and_log_eval_results_sklearn(results, config["setup"]["path_run"], test=True)
