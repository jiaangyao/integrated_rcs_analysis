import typing as tp

import ray
import numpy.typing as npt
import torch.nn as nn

from .base import TorchModelWrapper


class TorchMLPModel(nn.Module):
    def __init__(
            self,
            n_input,
            n_class,
            act_func,
            str_loss,
            n_layer=3,
            hidden_size=32,
            dropout=0.5,
    ):
        super().__init__()

        # initialize the fields
        self.n_input = n_input
        self.n_class = n_class
        self.n_layer = n_layer
        self.act_func = act_func
        self.str_loss = str_loss
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # define n_output based on the loss function
        if str_loss == "CrossEntropy":
            n_output = n_class
        else:
            n_output = 1

        # initialize the MLP layers
        if n_layer > 0:
            for i in range(n_layer):
                # append the linear layer
                if i == 0:
                    self.layers.append(nn.Linear(n_input, hidden_size))
                else:
                    self.layers.append(nn.Linear(hidden_size, hidden_size))

                # append the activation function
                self.layers.append(act_func)

                # append the dropout
                self.layers.append(nn.Dropout(self.dropout))

            # append the output layer
            self.layers.append(nn.Linear(hidden_size, n_output))

        else:
            # append the output layer only
            self.layers.append(nn.Linear(n_input, n_output))

            # append the activation function
            self.layers.append(act_func)

            # append the dropout
            self.layers.append(nn.Dropout(self.dropout))
        # sanity check
        assert len(self.layers) == 3 * self.n_layer + 1 if n_layer > 0 else 3

    def forward(self, x):
        if self.n_layer > 0:
            # forward pass
            for i in range(self.n_layer):
                x = self.layers[i](x)

            # output layer
            out = self.layers[-1](x)
        else:
            # output layer with dropout and activation
            x = self.layers[0](x)
            x = self.layers[1](x)
            out = self.layers[2](x)
        return out


# @ray.remote(num_gpus=0.125)
class MLPModelWrapper(TorchModelWrapper):
    def __init__(
        self,
        n_input,
        n_class,
        n_layer=3,
        hidden_size=32,
        dropout: float = 0.5,
        lam=1e-5,
        str_act="relu",
        str_reg="L2",
        str_loss="CrossEntropy",
        str_opt="Adam",
        lr=1e-4,
        transform=None,
        target_transform=None,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
    ):
        # initialize the base model
        super().__init__(
            n_class,
            str_loss,
            str_act,
            str_reg,
            lam,
            str_opt,
            lr,
            transform,
            target_transform,
            bool_use_ray=bool_use_ray,
            bool_use_gpu=bool_use_gpu,
            n_gpu_per_process=n_gpu_per_process,
        )

        # initialize fields for MLP params
        self.n_input = n_input
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lam = lam
        self.str_act = str_act
        self.str_reg = str_reg
        self.str_loss = str_loss
        self.str_opt = str_opt
        self.lr = lr
        self.transform = transform
        self.target_transform = target_transform
        self.bool_use_gpu = bool_use_gpu
        self.n_gpu_per_process = n_gpu_per_process

        # initialize the model
        self.model = TorchMLPModel(
            n_input,
            n_class,
            self.get_activation(str_act),
            str_loss,
            n_layer,
            hidden_size,
            dropout,
        )
        self.model = self.model.to(self.device)

    def train(
        self,
        train_data,
        train_label,
        valid_data: npt.NDArray | None = None,
        valid_label: npt.NDArray | None = None,
        batch_size: int = 32,
        n_epoch: int = 100,
        bool_early_stopping: bool = True,
        es_patience: int = 5,
        es_delta: float = 1e-5,
        es_metric: str = "val_loss",
        bool_shuffle: bool = True,
        bool_verbose: bool = True,
    ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
        # ensure that data is in the right format
        if train_data.ndim == 3:
            raise ValueError("Need to convert to 2D data first")

        vec_avg_loss, vec_avg_valid_loss = super().train(
            train_data,
            train_label,
            valid_data,
            valid_label,
            batch_size,
            n_epoch,
            bool_early_stopping,
            es_patience,
            es_delta,
            es_metric,
            bool_shuffle,
            bool_verbose,
        )

        return vec_avg_loss, vec_avg_valid_loss

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        # ensure that data is in the right format
        if data.ndim == 3:
            raise ValueError("Need to convert to 2D data first")

        return super().predict(data)


@ray.remote(num_cpus=1, num_gpus=0.2)
class MLPModelWrapperRay(MLPModelWrapper):
    def __init__(self, *args, **kwargs):
        # TODO: actor is deprecated, need to remove
        super().__init__(*args, **kwargs)