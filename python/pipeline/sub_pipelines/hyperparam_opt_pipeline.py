import wandb

def wandb_sweep_setup(eval, hyperparam_obj, data_class, config, logger):
    
    sweep_config = config["sweep"]
    wandb_config = config["wandb"]
    
    if (
        eval.model_type == "skorch"
        or eval.model_type == "torch"
    ):
        # Add input and output shape, which depends on data
        sweep_config["parameters"]["n_input"] = {"value": data_class.X_train.shape[-1]}
        # Assumes y is one-hot encoded
        # TODO: Remove this assumption
        sweep_config["parameters"]["n_class"] = {"value": data_class.y_train.shape[-1]}

    sweep_config["name"] = f"{config['run_name']}_{config['device']}_{config['time_stamp']}_sweep"

    sweep_id = wandb.sweep(
        sweep_config,
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
    )

    # Log relevant info
    logger.info("WandB project: {}".format(wandb_config.get('project')))
    logger.info("WandB entity: {}".format(wandb_config.get('entity')))
    logger.info(f"WandB sweep id: {sweep_id}")
    wandb_url = f"https://wandb.ai/{wandb_config.get('entity')}/{wandb_config.get('project')}/sweeps/{sweep_id}"
    logger.info(
        f"WandB sweep url: {wandb_url}"
    )
    logger.info("Local Directory Path: {}".format(config["run_dir"]))
    logger.info(f"WandB sweep config: {sweep_config}")
    hyperparam_obj.initialize_wandb_params(
        config["run_dir"], config["wandb"]["group"], config["wandb"]["tags"]
    )
    
    return sweep_id, sweep_config, wandb_url


def run_wandb_sweep(hyperparam_obj, sweep_method, num_runs, sweep_id):
    
    # Set-up agent
    wandb.agent(sweep_id, hyperparam_obj.wandb_sweep)
    
    # Run sweep
    if sweep_method == "grid":
        wandb.agent(sweep_id, function=hyperparam_obj.wandb_sweep)
    else:
        wandb.agent(
            sweep_id,
            function=hyperparam_obj.wandb_sweep,
            count=num_runs,
        )