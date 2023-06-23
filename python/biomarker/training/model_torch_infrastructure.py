# pyright: reportPrivateImportUsage=false
from __future__ import print_function
import typing as tp
from types import MappingProxyType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ray
import numpy as np
import numpy.typing as npt
from scipy.special import softmax  # type: ignore

import utils.torch_utils as ptu
from utils.torch_early_stopping import EarlyStopping
from biomarker.training.model_infrastructure import BaseModel
from biomarker.training.torch_dataset import NeuralDataset, NeuralDatasetTest


class PyTorchModelWrapper(BaseModel):
    def __init__(
        self,
        n_class,
        str_loss,
        str_act,
        str_reg,
        lam,
        str_opt,
        lr,
        transform,
        target_transform,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
    ):
        super().__init__()

        # initiate the model fields
        self.n_class = n_class

        # initialize the loss function
        self.str_loss = str_loss

        # initialize the activation function
        self.str_act = str_act

        # initialize the regularization
        self.str_reg = str_reg
        self.lam = lam

        # initialize the optimizer
        self.str_opt = str_opt
        self.lr = lr

        # initialize the transforms
        self.transform = transform
        self.target_transform = target_transform

        # initialize ray and GPU utilization flag
        self.bool_use_ray = bool_use_ray
        self.bool_use_gpu = bool_use_gpu

        # initialize GPU with memory constraints
        gpu_id = torch.cuda.current_device() if bool_use_gpu else 0
        bool_use_best_gpu = False if self.bool_use_ray else True
        bool_limit_gpu_mem = True if self.bool_use_ray else False
        ptu.init_gpu(
            use_gpu=bool_use_gpu,
            gpu_id=gpu_id,
            bool_use_best_gpu=bool_use_best_gpu,
            bool_limit_gpu_mem=bool_limit_gpu_mem,
            gpu_memory_fraction=n_gpu_per_process,
        )
        self.device = ptu.device

        # additional flags
        self.bool_torch = True

    def train(
        self,
        train_data,
        train_label,
        valid_data: npt.NDArray | None,
        valid_label: npt.NDArray | None,
        batch_size=32,
        n_epoch=100,
        bool_early_stopping=True,
        es_patience=5,
        es_delta=1e-5,
        bool_shuffle=True,
        bool_verbose=True,
    ):
        # double check device selection
        if self.bool_use_gpu:
            assert torch.cuda.is_available(), "Make sure you have a GPU available"
            assert self.device.type == "cuda", "Make sure you are using GPU"

            assert (
                torch.cuda.current_device() == self.device.index
            ), "Make sure you are using specified GPU"

        # initialize dataset and dataloader for training set
        train_dataset = NeuralDataset(
            train_data,
            train_label,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=bool_shuffle
        )

        # initialize the dataset and dataloader for validation set
        bool_run_validation = valid_data is not None and valid_label is not None
        valid_dataset = NeuralDataset(valid_data, valid_label)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=len(valid_dataset), shuffle=False
        )
        early_stopper = EarlyStopping(
            patience=es_patience,
            delta=es_delta,
            verbose=bool_verbose,
            bool_save_checkpoint=False,
        )

        # housekeeping and declare optimizer
        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()

        # train the model
        assert self.model is not None, "Make sure you have initialized the model"
        vec_avg_loss = []
        vec_avg_valid_loss = []
        for epoch in range(n_epoch):
            # initialize the loss and the model
            vec_loss = []
            vec_valid_loss = []
            self.model.train()

            # iterate through the dataset
            for idx_batch, (x, y) in enumerate(train_dataloader):
                assert (
                    self.model.training
                ), "make sure your network is in train mode with `.train()`"

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                y_pred = self.model(x)

                # compute the loss
                loss = loss_fn(y_pred, y)

                # backward pass
                loss.backward()
                optimizer.step()
                vec_loss.append(loss.item())

            # obtain the average loss
            loss = np.mean(np.stack(vec_loss, axis=0))
            vec_avg_loss.append(loss)

            # now perform validation
            str_valid_loss = None
            valid_loss = 0
            if bool_run_validation:
                # initialize the loss and set model to eval mode
                self.model.eval()  # type: ignore
                with torch.no_grad():
                    for _, (x_valid, y_valid) in enumerate(valid_dataloader):
                        assert (
                            not self.model.training
                        ), "make sure your network is in eval mode with `.eval()`"

                        # forward pass
                        y_valid_pred = self.model(x_valid)

                        # compute the loss
                        valid_loss = loss_fn(y_valid_pred, y_valid)
                        vec_valid_loss.append(valid_loss.item())

                    # obtain the average loss
                    valid_loss = np.mean(np.stack(vec_valid_loss, axis=0))
                    vec_avg_valid_loss.append(valid_loss)
                    str_valid_loss = ", Valid Loss: {:.4f}".format(valid_loss)

            # print the loss
            if bool_verbose:
                print(
                    "Epoch: {}, Loss: {:.4f}{}".format(epoch + 1, loss, str_valid_loss)
                )

            # call early stopping
            if bool_run_validation and bool_early_stopping:
                early_stopper(valid_loss, self.model)
                if early_stopper.early_stop:
                    if bool_verbose:
                        print("Early stopping")
                    break

        # load the last checkpoint with the best model
        self.model = early_stopper.load_model(self.model)
        vec_avg_loss = np.stack(vec_avg_loss, axis=0)
        vec_avg_valid_loss = np.stack(vec_avg_valid_loss, axis=0)

        return vec_avg_loss, vec_avg_valid_loss

    def predict(self, data: np.ndarray):
        # initialize the dataset and dataloader for testing set
        test_dataset = NeuralDatasetTest(data)
        test_dataloader = DataLoader(
            test_dataset, batch_size=data.shape[0], shuffle=False
        )

        # initialize the loss and set model to eval mode
        self.model.eval()
        vec_y_pred = []
        with torch.no_grad():
            for _, x_test in enumerate(test_dataloader):
                assert (
                    not self.model.training
                ), "make sure your network is in eval mode with `.eval()`"

                # forward pass
                y_test_pred = self.model(x_test)

                # append the probability
                vec_y_pred.append(y_test_pred)

        # stack the prediction
        y_pred = torch.cat(vec_y_pred, dim=0)

        return ptu.to_numpy(y_pred)

    def get_accuracy(self, data: np.ndarray, label: np.ndarray):
        # obtain the prediction
        y_pred = np.argmax(self.predict(data), axis=1)
        y_pred = self._check_input(y_pred)

        # if label is one-hot encoded, convert it to integer
        y_real = self._check_input(label)

        return np.sum(y_pred == y_real) / y_real.shape[0]

    def predict_proba(self, data):
        class_prob = softmax(self.predict(data), axis=1)

        if class_prob.shape[1] == 2:
            class_prob = class_prob[:, 1]

        return class_prob

    def get_loss(self):
        if self.str_loss == "CrossEntropy":
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        return loss

    def get_activation(self, str_act):
        act_hash_map = ptu.get_act_func()
        if str_act in act_hash_map.keys():
            act = act_hash_map[str_act]
        else:
            raise NotImplementedError

        return act

    def get_regularizer(self):
        if self.str_reg == "L2":
            if self.lam != 0:
                reg_params = {"weight_decay": self.lam}
            else:
                reg_params = {}
        else:
            raise NotImplementedError

        return reg_params

    def get_optimizer(self):
        # obtain the regularizer
        reg_params = self.get_regularizer()

        # initialize the optimizer
        if self.str_opt == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, **reg_params
            )
        else:
            raise NotImplementedError

        return optimizer


class TorchMLPModel(nn.Module):
    def __init__(
        self,
        n_input,
        n_class,
        act_func,
        loss_func,
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
        self.loss_func = loss_func
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # initialize the MLP layers
        if n_layer > 0:
            for i in range(n_layer):
                if i == 0:
                    self.layers.append(nn.Linear(n_input, hidden_size))
                else:
                    self.layers.append(nn.Linear(hidden_size, hidden_size))

            # append the output layer
            self.layers.append(nn.Linear(hidden_size, n_class))
        else:
            self.layers.append(nn.Linear(n_input, n_class))

    def forward(self, x):
        # forward pass
        for i in range(self.n_layer):
            x = self.layers[i](x)
            x = self.act_func(x)
            x = nn.Dropout(self.dropout)(x)

        # output layer
        out = self.layers[-1](x)

        return out


class TorchRNNModel(nn.Module):
    def __init__(
        self,
        n_input,
        n_class,
        loss_func,
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
        self.loss_func = loss_func

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
        self.output = nn.Linear(self.rnn_dim * self.multi, self.n_class)
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


# @ray.remote(num_gpus=0.125)
class MLPModelWrapper(PyTorchModelWrapper):
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
            self.get_activation(self.str_act),
            self.get_loss(),
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


class RNNModelWrapper(PyTorchModelWrapper):
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
            self.get_loss(),
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
        valid_data: tp.Optional[npt.NDArray] = None,
        valid_label: tp.Optional[npt.NDArray] = None,
        batch_size: int = 32,
        n_epoch: int = 100,
        bool_early_stopping: bool = True,
        es_patience: int = 5,
        es_delta: float = 1e-5,
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
