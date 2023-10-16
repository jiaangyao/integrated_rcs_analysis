# pyright: reportPrivateImportUsage=false
from __future__ import print_function
import typing as tp
from types import MappingProxyType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torchmetrics

import numpy as np
import numpy.typing as npt
from scipy.special import softmax  # type: ignore

import model.torch_model.torch_utils as ptu
from dataset.torch_dataset import NeuralDataset, NeuralDatasetTest
from ..base import BaseModel
from .callbacks import EarlyStopping


class TorchModelWrapper(BaseModel):
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

        # initialize the model fields
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
        es_metric="val_loss",
        bool_shuffle=True,
        bool_verbose=True,
    ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
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

        # define task for accuracy metric
        if self.n_class == 2:
            str_task = "binary"
        else:
            str_task = "multiclass"

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
            vec_acc = []
            vec_valid_loss = []
            vec_valid_acc = []
            self.model.train()

            # iterate through the dataset
            for idx_batch, (x_train, y) in enumerate(train_dataloader):
                assert (
                    self.model.training
                ), "make sure your network is in train mode with `.train()`"

                # forward pass
                y_pred = self.model(x_train)

                # compute the loss
                loss = loss_fn(y_pred, y)

                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # append loss
                vec_loss.append(loss.item())

                # compute the accuracy
                # noinspection PyTypeChecker
                acc = torchmetrics.functional.accuracy(
                    torch.argmax(y_pred, dim=1), y, task=str_task
                )
                vec_acc.append(acc.item())

            # obtain the average loss
            loss = np.mean(np.stack(vec_loss, axis=0))
            acc = np.mean(np.stack(vec_acc, axis=0))
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

                        # compute the accuracy
                        # noinspection PyTypeChecker
                        valid_acc = torchmetrics.functional.accuracy(
                            torch.argmax(y_valid_pred, dim=1), y_valid, task=str_task
                        )
                        vec_valid_acc.append(valid_acc.item())

                    # obtain the average loss
                    valid_loss = np.mean(np.stack(vec_valid_loss, axis=0))
                    valid_acc = np.mean(np.stack(vec_valid_acc, axis=0))
                    vec_avg_valid_loss.append(valid_loss)
                    str_valid_loss = ", Valid Loss: {:.4f}".format(valid_loss)
                    str_valid_acc = ", Valid Acc: {:.4f}".format(valid_acc)

            # print the loss
            if bool_verbose:
                print(
                    "Epoch: {}, Loss: {:.4f}{}".format(epoch + 1, loss, str_valid_loss)
                )
                print("Epoch: {}, Acc: {:.4f}{}".format(epoch + 1, acc, str_valid_acc))

            # call early stopping
            if bool_run_validation and bool_early_stopping:
                if es_metric == "val_loss":
                    early_stopper(valid_loss, self.model)
                elif es_metric == "val_acc":
                    early_stopper(valid_acc, self.model, mode="max")
                else:
                    raise ValueError("es_metric must be either 'val_loss' or 'val_acc'")

                if early_stopper.early_stop:
                    if bool_verbose:
                        print("Early stopping")
                    break

        # load the last checkpoint with the best model
        self.model = early_stopper.load_model(self.model)
        vec_avg_loss = np.stack(vec_avg_loss, axis=0)
        vec_avg_valid_loss = np.stack(vec_avg_valid_loss, axis=0)

        return vec_avg_loss, vec_avg_valid_loss

    def predict(self, data: npt.NDArray) -> npt.NDArray:
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

    def get_accuracy(
        self,
        data: npt.NDArray,
        label: npt.NDArray,
    ):
        # obtain the prediction
        y_pred = np.argmax(self.predict(data), axis=1)
        y_pred = self._check_input(y_pred)

        # if label is one-hot encoded, convert it to integer
        y_real = self._check_input(label)

        return np.sum(y_pred == y_real) / y_real.shape[0]

    def predict_proba(self, data):
        class_prob = softmax(self.predict(data), axis=1)

        # if class_prob.shape[1] == 2:
        #     class_prob = class_prob[:, 1]

        return class_prob

    def get_auc(
        self,
        scores: npt.NDArray,
        label: npt.NDArray,
    ) -> npt.NDArray:
        auc = torchmetrics.AUROC(task="multiclass", num_classes=self.n_class)(
            torch.Tensor(scores), torch.Tensor(label).to(torch.long)
        )

        return ptu.to_numpy(auc)

    def get_loss(self):
        if self.str_loss == "CrossEntropy":
            loss = torch.nn.CrossEntropyLoss()
        elif self.str_loss == "BCE":
            loss = torch.nn.BCELoss()
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
