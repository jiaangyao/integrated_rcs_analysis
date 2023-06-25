from types import MappingProxyType

import typing as tp
import numpy as np
import numpy.typing as npt
from hydra import compose
from omegaconf import DictConfig, OmegaConf

from biomarker.training.model_infrastructure import *
from biomarker.training.model_torch_infrastructure import (
    MLPModelWrapper,
    RNNModelWrapper,
)

_VEC_MODEL_DYNAMICS_ONLY = ["RNN"]


def get_model_params(
    str_model: str,
    bool_use_ray: bool = False,
    bool_use_gpu: bool = False,
    n_gpu_per_process: int | float = 0,
    bool_tune_hyperparams: bool = False,
):

    # load the model parameters
    if not bool_tune_hyperparams:
        # if not tuning hyperparameters, then load default config file
        model_cfg = compose(
            config_name="model_config", overrides=["model=default_{}".format(str_model)]
        )["model"]
    else:
        # if tuning hyperparameters, then load the tuning config file
        model_cfg = compose(
            config_name="model_config", overrides=["model=tune_{}".format(str_model)]
        )["model"]
    assert model_cfg is not None, "Model config must be initialized"

    # load the trainer parameters
    train_cfg = compose(
        config_name="trainer_config",
        overrides=["trainer={}_trainer".format(model_cfg["model_type"])],
    )["trainer"]

    # update the GPU per process in case updated during initialization
    model_cfg["bool_use_gpu"] = bool_use_gpu
    if model_cfg["model_type"] == "torch":
        model_cfg["model_kwargs"]["bool_use_ray"] = bool_use_ray
        model_cfg["model_kwargs"]["bool_use_gpu"] = bool_use_gpu
        model_cfg["model_kwargs"]["n_gpu_per_process"] = float(n_gpu_per_process)

    return model_cfg, train_cfg


def get_dynamics_model() -> list:
    return _VEC_MODEL_DYNAMICS_ONLY


def initialize_model_sklearn(
    str_model: str,
    model_args: list | tuple = tuple(),
    model_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
    ensemble_args: list | tuple = tuple(),
    ensemble_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
):
    # TODO: avoid importing all models via *

    if str_model == "LDA":
        model = LDAModel(model_args, model_kwargs)
    elif str_model == "QDA":
        model = QDAModel(model_args, model_kwargs)
    elif str_model == "SVM":
        model = SVMModel(model_args, model_kwargs)
    elif str_model == "RF":
        model = RandomForestModel(model_args, model_kwargs)
    elif str_model == "AdaBoost":
        model = AdaBoostModel(model_args, model_kwargs, ensemble_args, ensemble_kwargs)
    elif str_model == "GP":
        model = GPModel(model_args, model_kwargs)
    elif str_model == "GBoost":
        model = GradientBoostingModel(model_args, model_kwargs)
    else:
        raise NotImplementedError

    return model


def initialize_model_torch(
    str_model: str,
    model_args: list | tuple = tuple(),
    model_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
):
    # TODO: avoid importing all models via *

    if str_model == "MLP":
        # initialize the model
        model = MLPModelWrapper(
            *model_args,
            **model_kwargs,
        )

    elif str_model == "RNN":
        # initalize the model
        model = RNNModelWrapper(
            *model_args,
            **model_kwargs,
        )
    else:
        raise NotImplementedError

    return model


def initialize_model(
    str_model: str,
    model_cfg: DictConfig | dict,
    n_input: int,
    n_class_model: int,
):
    # now check if using torch models
    bool_torch = model_cfg["model_type"] == "torch"

    # now define the model
    # if using pytorch models
    if bool_torch:
        # append correct dimensionality to model args
        model_cfg["model_args"] = [n_input, n_class_model]

        # now start the model training
        model = initialize_model_torch(
            str_model, model_cfg["model_args"], model_cfg["model_kwargs"]
        )

    # if using sklearn models
    else:
        # warn if using GPU for sklearn
        if model_cfg["bool_use_gpu"]:
            Warning("Using GPU for sklearn model training")

        # now obtain the models
        model = initialize_model_sklearn(
            str_model,
            model_cfg["model_args"],
            model_cfg["model_kwargs"],
            model_cfg["ensemble_args"],
            model_cfg["ensemble_kwargs"],
        )

    return model
