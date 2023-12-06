import wandb

# Libraries for hyperparameter tuning
from training_eval.hyperparameter_optimization import HyperparameterOptimization


def wandb_sweep_setup(eval, hyperparam_obj, data_class, config, logger):

    sweep_config = config["sweep"]
    setup_config = config["setup"]
    wandb_config = setup_config["wandb"]

    if eval.model_type == "skorch" or eval.model_type == "torch":
        # Add input and output shape, which depends on data
        sweep_config["parameters"]["n_input"] = {"value": data_class.X_train.shape[-1]}
        # Assumes y is one-hot encoded
        # TODO: Remove this assumption
        sweep_config["parameters"]["n_class"] = {"value": data_class.y_train.shape[-1]}

    # TODO: Debug name... throws error
    # sweep_config[
    #     "name"
    # ] = f"{setup_config['run_name']}_{setup_config['device']}_{setup_config['time_stamp']}_sweep"

    sweep_id = wandb.sweep(
        sweep_config,
        project=wandb_config["project"],
        entity=wandb_config["entity"],
    )

    # Log relevant info
    logger.info("WandB project: {}".format(wandb_config.get("project")))
    logger.info("WandB entity: {}".format(wandb_config.get("entity")))
    logger.info(f"WandB sweep id: {sweep_id}")
    wandb_url = f"https://wandb.ai/{wandb_config.get('entity')}/{wandb_config.get('project')}/sweeps/{sweep_id}"
    logger.info(f"WandB sweep url: {wandb_url}")
    logger.info("Local Directory Path: {}".format(setup_config["path_run"]))
    logger.info(f"WandB sweep config: {sweep_config}")
    hyperparam_obj.initialize_wandb_params(
        setup_config["path_run"], wandb_config["group"], wandb_config["tags"]
    )

    return sweep_id, sweep_config, wandb_url


def run_wandb_sweep(hyperparam_obj, sweep_method, num_runs, sweep_id):

    # Run sweep
    if sweep_method == "grid":
        wandb.agent(sweep_id, function=hyperparam_obj.wandb_sweep)
    else:
        wandb.agent(
            sweep_id,
            function=hyperparam_obj.wandb_sweep,
            count=num_runs,
        )


def run_hyperparameter_search(config, model_class, data, eval, logger):
    
    # Hyperparameter search. First, check which library to use
    ho = HyperparameterOptimization(model_class, data, eval)
    
    # Check if hyperparameter search is desired
    if (hyperparam_config := config.get('hyperparameter_optimization')) is None:
        # If not, train and evaluate model with default hyperparameters
        ho.train_and_eval_no_search(config)
        return None, None

    # Run hyperparameter search using WandB sweeps
    elif hyperparam_config["search_library"].lower() == "wandb":
        (
            sweep_id,
            sweep_config,
            sweep_url,
        ) = wandb_sweep_setup(
            eval, ho, data, config, logger
        )
        
        run_wandb_sweep(
            ho, sweep_config["method"], hyperparam_config["num_runs"], sweep_id
        )
        
        return sweep_url, sweep_id

    # Run hyperparameter search using Optuna
    elif hyperparam_config["search_library"].lower() == "optuna":
        raise NotImplementedError
    