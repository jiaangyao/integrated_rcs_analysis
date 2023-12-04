import os
from omegaconf import OmegaConf, DictConfig
import wandb
from loguru import logger

from utils.file_utils import create_zip, get_git_info, add_config_to_csv, save_conda_package_versions

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
    # Save code, git info, and config file to run directory
    create_zip(f'{os.getcwd()}/python', f'{config["run_dir"]}/code.zip', exclude=config["code_snapshot_exlude"])
    save_conda_package_versions(config["run_dir"])
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
            notes=cfg.wandb.notes,
            dir=config["run_dir"],
        )
        logger.info("WandB run url: {}".format(run.url))
        logger.info("WandB project: {}".format(run.project))
        logger.info("WandB entity: {}".format(run.entity))
        logger.info("WandB run name: {}".format(run.name))
        logger.info("WandB run id: {}".format(run.id))
        logger.info("Local Directory Path: {}".format(config["run_dir"]))
        wandb.log({'metadata/local_dir': config["run_dir"]})

    return config, logger