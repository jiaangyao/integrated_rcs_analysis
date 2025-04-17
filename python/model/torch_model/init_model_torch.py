from types import MappingProxyType

from .callbacks import EarlyStopping

# from .mlp_model import MLPModelWrapper
from .rnn_model import RNNModelWrapper


def init_model_torch(
    str_model: str,
    model_args: list | tuple = tuple(),
    model_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
):
    # TODO: avoid importing all models via *

    if str_model == "MLP":
        # initialize the model
        # model = MLPModelWrapper(
        #     *model_args,
        #     **model_kwargs,
        # )
        # TODO: Update with new MLPModel class
        raise NotImplementedError

    elif str_model == "RNN":
        # initialize the model
        model = RNNModelWrapper(
            *model_args,
            **model_kwargs,
        )
    else:
        raise NotImplementedError

    return model
