import numpy.typing as npt
from hydra import compose

# TODO: move all constants to constants directory
_VEC_MODEL_DYNAMICS_ONLY = ["RNN"]


class BaseModel:
    """Base model class to be inherited by all sklearn and pytorch models
    """    
    def __init__(self):
        self.model = None

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _check_input(data: npt.NDArray):
        if len(data.shape) == 1:
            data = data[:, None]
        return data


def get_dynamics_model() -> list:
    return _VEC_MODEL_DYNAMICS_ONLY
