import os
from omegaconf import OmegaConf, DictConfig
import wandb
from loguru import logger

from utils.file_utils import (
    create_zip,
    get_git_info,
    save_conda_package_versions,
)


def setup(config, WandB_hyperopt=False):
    logger.add(
        config["file_log"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
        level="INFO",
    )
    # Save code, git info, and config file to run directory
    create_zip(
        f"{os.getcwd()}/python",
        f'{config["path_run"]}/code.zip',
        exclude=config["code_snapshot_exlude"],
    )
    save_conda_package_versions(config["path_run"])
    git_info = get_git_info()
    logger.info("Git info: {}".format(git_info))
    config |= git_info

    logger.info(f"Beginning pipeline...")

    # 2. Log config file to wandb, set up hydra logging, and save to disk
    if (
        (wandb_setup := config.get("wandb")) is not None
        and not WandB_hyperopt
    ):
        wandb.config = config
        run = wandb.init(
            entity=wandb_setup.get("entity"),
            project=wandb_setup.get("project"),
            group=wandb_setup.get("group"),
            tags=wandb_setup.get("tags"),
            notes=wandb_setup.get("notes"),
            dir=config.get("path_run"),
        )
        logger.info("WandB run url: {}".format(run.url))
        logger.info("WandB project: {}".format(run.project))
        logger.info("WandB entity: {}".format(run.entity))
        logger.info("WandB run name: {}".format(run.name))
        logger.info("WandB run id: {}".format(run.id))
        logger.info("Local Directory Path: {}".format(config["path_run"]))
        wandb.log({"metadata/local_dir": config.get("path_run")})

    return logger
