import typing as tp

import ray
import numpy.typing as npt
import torch.nn as nn

# TODO: Below caused circular import error with torch_model.base, need to fix
# from .base import TorchModelWrapper


class TorchRNNModel(nn.Module):
    def __init__(
        self,
        n_input,
        n_class,
        str_loss,
        bool_cnn=True,
        cnn_act_func=None,
        cnn_ks=5,
        cnn_stride=3,
        bool_cnn_dropout=False,
        cnn_dropout=0.5,
        n_rnn_layer=2,
        rnn_dim=32,
        bool_bidirectional=True,
        rnn_dropout=0.5,
        bool_final_dropout=True,
        final_dropout=0.5,
    ):
        super().__init__()

        # initialize the fields
        self.n_input = n_input
        self.n_class = n_class
        self.str_loss = str_loss

        # define n_output based on the loss function
        if str_loss == "CrossEntropy":
            n_output = n_class
        else:
            n_output = 1

        # cnn preprocessing layer parameters
        self.bool_cnn = bool_cnn
        self.cnn_act_func = cnn_act_func
        self.cnn_ks = cnn_ks
        self.cnn_stride = cnn_stride
        self.bool_cnn_dropout = bool_cnn_dropout
        self.cnn_dropout = cnn_dropout

        # rnn layer parameters
        self.n_rnn_layer = n_rnn_layer
        self.rnn_dim = rnn_dim
        self.bool_bidirectional = bool_bidirectional
        self.multi = 2 if bool_bidirectional else 1
        self.rnn_dropout = rnn_dropout

        # output layer parameters
        self.bool_final_dropout = bool_final_dropout
        self.final_dropout = final_dropout

        # initialize the CNN layers
        self.conv1d = nn.Conv1d(
            in_channels=n_input,
            out_channels=self.rnn_dim,
            kernel_size=self.cnn_ks,
            stride=self.cnn_stride,
        )
        self.cnn_dropout_layer = nn.Dropout(self.cnn_dropout)

        # initialize the RNN layers
        input_size_rnn = n_input if not self.bool_cnn else self.rnn_dim
        self.biGRU = nn.GRU(
            input_size=input_size_rnn,
            hidden_size=self.rnn_dim,
            num_layers=self.n_rnn_layer,
            bidirectional=self.bool_bidirectional,
            dropout=self.rnn_dropout,
        )

        # initialize the final output layer
        self.output = nn.Linear(self.rnn_dim * self.multi, n_output)
        self.final_dropout_layer = nn.Dropout(self.final_dropout)

    def forward(self, x):
        # forward pass
        # if choose to optionally use CNN preprocessing
        # input data in format (batch_size, n_channel, n_time)
        if self.bool_cnn:
            x = self.conv1d(x)  # preprocessing kernel in time
            if self.cnn_act_func is not None:
                x = self.cnn_act_func(x)
            if self.bool_cnn_dropout:
                x = self.cnn_dropout_layer(x)

            x = x.permute(2, 0, 1)  # now in format (n_time, batch_size, n_channel)
        else:
            x = x.permute(2, 0, 1)  # now in format (n_time, batch_size, n_channel)

        # RNN processing - need hidden state for classification
        _, x = self.biGRU(x)
        x = x.contiguous()[
            -self.multi :, :, :
        ]  # now shape (self.multi, batch_size, rnn_dim)

        # output layer
        x = x.permute(1, 0, 2)  # now in format (batch_size, self.multi, rnn_dim)
        x = x.contiguous().view(
            -1, self.rnn_dim * self.multi
        )  # now in format (batch_size, self.multi * rnn_dim)
        if self.bool_final_dropout:
            x = self.final_dropout_layer(x)
        out = self.output(x)

        return out


# class RNNModelWrapper(TorchModelWrapper):
class RNNModelWrapper:
    def __init__(
        self,
        n_input,
        n_class,
        bool_cnn=True,
        str_cnn_act="identity",
        cnn_ks=5,
        cnn_stride=3,
        bool_cnn_dropout=False,
        cnn_dropout=0.5,
        n_rnn_layer=2,
        rnn_dim=32,
        bool_bidirectional=True,
        rnn_dropout=0.5,
        bool_final_dropout=True,
        final_dropout=0.5,
        lam=1e-5,
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
            str_cnn_act,
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

        # initialize fields for RNN params
        self.n_input = n_input
        self.bool_cnn = bool_cnn
        self.str_cnn_act = str_cnn_act
        self.cnn_ks = cnn_ks
        self.cnn_stride = cnn_stride
        self.bool_cnn_dropout = bool_cnn_dropout
        self.cnn_dropout = cnn_dropout
        self.n_rnn_layer = n_rnn_layer
        self.rnn_dim = rnn_dim
        self.bool_bidirectional = bool_bidirectional
        self.rnn_dropout = rnn_dropout
        self.bool_final_dropout = bool_final_dropout
        self.final_dropout = final_dropout
        self.lam = lam
        self.str_reg = str_reg
        self.str_loss = str_loss
        self.str_opt = str_opt
        self.lr = lr
        self.transform = transform
        self.target_transform = target_transform
        self.bool_use_gpu = bool_use_gpu
        self.n_gpu_per_process = n_gpu_per_process

        # initialize the model
        self.model = TorchRNNModel(
            n_input,
            n_class,
            str_loss,
            bool_cnn,
            self.get_activation(self.str_cnn_act),
            cnn_ks,
            cnn_stride,
            bool_cnn_dropout,
            cnn_dropout,
            n_rnn_layer,
            rnn_dim,
            bool_bidirectional,
            rnn_dropout,
            bool_final_dropout,
            final_dropout,
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
        if train_data.ndim != 3:
            raise ValueError("Has to be 3D data for RNN model")

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
        if data.ndim != 3:
            raise ValueError("Has to be 3D data for RNN model")

        return super().predict(data)


@ray.remote(num_cpus=1, num_gpus=0.2)
class RNNModelWrapperRay(RNNModelWrapper):
    def __init__(self, *args, **kwargs):
        # TODO: actor is deprecated, need to remove
        super().__init__(*args, **kwargs)
