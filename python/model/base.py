import numpy.typing as npt
from hydra import compose
from training_eval.model_evaluation import *
import wandb
import numpy as np

# TODO: move all constants to constants directory
_VEC_MODEL_DYNAMICS_ONLY = ["RNN"]


class BaseModel:
    """Base model class to be inherited by all sklearn and pytorch models"""

    def __init__(self, model):
        self.model = model

    def predict(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def override_model(self, model_args, model_kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _check_input(data: npt.NDArray):
        if len(data.shape) == 1:
            data = data[:, None]
        return data


class ArbitraryModel(BaseModel):
    """
    This class can be used to wrap any model, then compare in a pipeline with other models
    """

    def __init__(self):
        super().__init__()


def get_dynamics_model() -> list:
    return _VEC_MODEL_DYNAMICS_ONLY
