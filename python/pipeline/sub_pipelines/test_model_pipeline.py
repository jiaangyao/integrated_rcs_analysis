import wandb

def get_model(model_class, model_options, logger):
    """
    Returns the model class, either by loading the model from a file, 
    resetting the model, or creating a model from the model options.
    """
    if model_class is None:
        raise ValueError("Model class is None.")
    
    model_instantiation = model_options.get("model_instantiation")
    
    if model_instantiation == "reset" or model_instantiation == "default":
        logger.info("Resetting model for testing.")
        model_class = model_class.reset_model()
        
    elif model_instantiation == "load":
        logger.info("Loading model.")
        raise NotImplementedError("Loading models is not implemented yet.")
    
    elif model_instantiation == "from_WandB_sweep":
        
        logger.info("Creating model from best hyperparameters from sweep.")
        
        # Create model from best hyperparameters
        model_class = model_class.override_model(model_options.get("best_run_config"))
    
    else:
        raise ValueError(f"Please choose a supported model instantiation option.")


def test_model(model_class, eval, data, config, logger):
    """
    Evaluates the model on the test set, and logs the results to WandB.
    """
    # Reset model, load model, or create model class from config generated in hyperparameter search
    model_class = get_model(model_class, config.get("model_options"), logger)
    
    # Train Model
    model_class.train(data)
    
    
    # Evaluate model
    
    
    # Log results (to WandB)
    
    
    # Save model (if specified)