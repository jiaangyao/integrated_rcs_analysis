# pyright: reportPrivateImportUsage=false
from __future__ import print_function
import typing as tp
from types import MappingProxyType
from typing import Any
import torchmetrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics

import numpy as np
import numpy.typing as npt
from scipy.special import softmax  # type: ignore

import model.torch_model.torch_utils as ptu
from dataset.torch_dataset import NeuralDataset, NeuralDatasetTest
from ..base import BaseModel
#from .callbacks import EarlyStopping
#from .mlp_model import MLPModelWrapper
from .rnn_model import RNNModelWrapper
from sklearn.model_selection import train_test_split

# TODO: transfer constants to constants directory
# define global variables
# _STR_TO_ACTIVATION = {
#     "relu": nn.ReLU(),
#     "tanh": nn.Tanh(),
#     "leaky_relu": nn.LeakyReLU(),
#     "sigmoid": nn.Sigmoid(),
#     "selu": nn.SELU(),
#     "softplus": nn.Softplus(),
#     "identity": nn.Identity(),
# }


class BaseTorchClassifier:
    def __init__() -> None:
        pass
    
    # TODO: Move predict and get_accuracy elsewhere, as they call model but are not part of model architecture
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

    # TODO: Move get_accuracy and get_auc to Evaluation class??
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
    

    def get_activation(self, str_act):
        # act_hash_map = ptu.get_act_func()
        # if str_act in act_hash_map.keys():
        #     act = act_hash_map[str_act]
        # else:
        #     raise NotImplementedError

        # return act
        return ptu.get_act_func(str_act)

# TODO: Reduce complexity in train() call
# TODO: This could be achieved by handling early stopping and GPU utilization in separate classes or locations
# TODO: Probably shouldn't calculate accuracy in train() call, move to evaluation class
class BaseTorchTrainer():
    def __init__(
        self,
        model,
        early_stopping,
        n_class,
        str_loss,
        str_reg,
        lam,
        str_opt,
        lr,
        transform,
        target_transform,
        batch_size=32,
        n_epoch=100,
        bool_shuffle=True,
        bool_verbose=False,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
    ):

        # initialize the model fields
        self.model = model
        self.n_class = n_class

        # initialize the loss function
        self.str_loss = str_loss

        # initialize the regularization
        self.str_reg = str_reg
        self.lam = lam

        # initialize the optimizer
        self.str_opt = str_opt
        self.lr = lr

        # initialize the transforms
        self.transform = transform
        self.target_transform = target_transform
        
        # Initialize trainer data parameters
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.bool_shuffle = bool_shuffle
        self.bool_verbose = bool_verbose
        
        # Initialize early stopper
        self.early_stopping = early_stopping
        
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
        # Removing validation data for now, as it is not used if early stopping is not enabled. Split occurs if ealry stopping is enabled
        # valid_data: npt.NDArray | None,
        # valid_label: npt.NDArray | None,
    ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
        
        # TODO: Flag to clear model params before training?? This would avoid stacking multiple training sessions on top of each other
        # TODO: Remove accuracy calculation from training loop, move to evaluation class
        # double check device selection
        if self.bool_use_gpu:
            assert torch.cuda.is_available(), "Make sure you have a GPU available"
            assert self.device.type == "cuda", "Make sure you are using GPU"

            assert (
                torch.cuda.current_device() == self.device.index
            ), "Make sure you are using specified GPU"

        if self.early_stopping is not None:
            # ! Currently, validation set is only created if early stopping is enabled.
            train_data, valid_data, train_label, valid_label = train_test_split(
                train_data, train_label, test_size=self.early_stopping.validation_split, random_state=self.early_stopping.random_seed
            )
            
            valid_dataset = NeuralDataset(valid_data, valid_label)
            valid_dataloader = DataLoader(
                valid_dataset, batch_size=len(valid_dataset), shuffle=False
            )
        
        # initialize dataset and dataloader for training set
        train_dataset = NeuralDataset(
            train_data,
            train_label,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle
        )

        # define task for accuracy metric
        if self.n_class == 2:
            str_task = "binary"
        else:
            str_task = "multiclass"

        # initialize the dataset and dataloader for validation set
        # TODO: Fix bool_run_validation
        # bool_run_validation = valid_data is not None and valid_label is not None
        bool_run_validation = False

        # housekeeping and declare optimizer
        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()

        # train the model
        assert self.model is not None, "Make sure you have initialized the model"
        vec_avg_loss = []
        vec_avg_valid_loss = []
        for epoch in range(self.n_epoch):
            # initialize the loss and the model
            vec_loss = []
            vec_acc = []
            self.model.train()

            # iterate through the dataset
            for idx_batch, (x_train, y) in enumerate(train_dataloader):
                assert (
                    self.model.training
                ), "make sure your network is in train mode with `.train()`"

                # forward pass
                y_pred = self.model(x_train)

                # compute the loss
                loss = loss_fn(y_pred, y.float())

                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # append loss
                vec_loss.append(loss.item())

                # compute the accuracy
                # noinspection PyTypeChecker
                # TODO: Consider moving accuracy calculation to evaluation class
                acc = torchmetrics.functional.accuracy(
                    torch.argmax(y_pred, dim=1), torch.argmax(y, dim=1), task=str_task, num_classes=self.n_class
                )
                vec_acc.append(acc.item())

            # obtain the average loss
            loss = np.mean(np.stack(vec_loss, axis=0))
            acc = np.mean(np.stack(vec_acc, axis=0))
            vec_avg_loss.append(loss)

            # now perform validation
            # TODO: Consider consolidating validation with early stopping
            # TODO: Consider moving early stopping check to evaluation class (e.g. early_stopping_check() or early_stopping_criteria_met() )
            str_valid_loss = None
            valid_loss = 0
            if bool_run_validation and self.early_stopping:
                # initialize the loss and set model to eval mode
                vec_valid_loss = []
                vec_valid_acc = []
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
                            torch.argmax(y_valid_pred, dim=1), torch.argmax(y_valid, dim=1), task=str_task, num_classes=self.n_class
                        )
                        vec_valid_acc.append(valid_acc.item())

                    # obtain the average loss
                    valid_loss = np.mean(np.stack(vec_valid_loss, axis=0))
                    valid_acc = np.mean(np.stack(vec_valid_acc, axis=0))
                    vec_avg_valid_loss.append(valid_loss)
                    str_valid_loss = ", Valid Loss: {:.4f}".format(valid_loss)
                    str_valid_acc = ", Valid Acc: {:.4f}".format(valid_acc)

                # print the loss
                if self.bool_verbose:
                    print(
                        "Epoch: {}, Loss: {:.4f}{}".format(epoch + 1, loss, str_valid_loss)
                    )
                    print("Epoch: {}, Acc: {:.4f}{}".format(epoch + 1, acc, str_valid_acc))

            # call early stopping
            if bool_run_validation and self.early_stopping:
                if self.es_metric == "val_loss":
                    self.early_stopping(valid_loss, self.model)
                elif self.es_metric == "val_acc":
                    self.early_stopping(valid_acc, self.model, mode="max")
                else:
                    raise ValueError("es_metric must be either 'val_loss' or 'val_acc'")

                if self.early_stopping.early_stop:
                    if self.bool_verbose:
                        print("Early stopping")
                    break

                # load the last checkpoint with the best model
                self.model = self.early_stopping.load_model(self.model)
                
        vec_avg_loss = np.stack(vec_avg_loss, axis=0)
        if bool_run_validation:
            vec_avg_valid_loss = np.stack(vec_avg_valid_loss, axis=0)
        else:
            vec_avg_valid_loss = np.array([])

        return vec_avg_loss, vec_avg_valid_loss


    def get_loss(self):
        if self.str_loss == "CrossEntropy":
            loss = torch.nn.CrossEntropyLoss()
        elif self.str_loss == "BCE":
            loss = torch.nn.BCELoss()
        elif self.str_loss == "MSE":
            loss = torch.nn.MSELoss()
        else:
            raise ValueError(f'Criterion {self.str_loss} not supported')

        return loss


    def get_optimizer(self):
        # obtain the regularizer
        reg_params = self.get_regularizer()

        # initialize the optimizer
        if isinstance(self.str_opt, optim.Optimizer):
            return optimizer
        
        elif self.str_opt.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, **reg_params
            )
        # initialize the optimizer
        elif self.str_opt.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, **reg_params
            )
        else:
            raise ValueError(f'Optimizer {self.str_opt} not supported')

        return optimizer
    
    
    def get_regularizer(self):
        if self.str_reg == "L2":
            if self.lam != 0:
                reg_params = {"weight_decay": self.lam}
            else:
                reg_params = {}
        else:
            raise NotImplementedError

        return reg_params


class BaseTorchModel(BaseModel):
    
    MODEL_ARCHITECURE_KEYS = ["n_input", "n_class", "act_func", "n_layer", "hidden_size", "act_func", "dropout"]
    TRAINER_KEYS = [
        "n_class", "str_loss", "str_reg", "lam", "str_opt", "lr", "transform", "target_transform",
        "batch_size", "n_epoch", "bool_shuffle", "bool_verbose", "bool_use_ray", "bool_use_gpu", "n_gpu_per_process", "criterion", "optimizer",
        "learning_rate"
    ]
    EARLYSTOPPING_KEYS = ["bool_early_stopping", "es_patience", "es_delta",
        "es_metric", "bool_verbose"]
    
    def __init__(self, model, trainer, early_stopping=None, model_kwargs=None, trainer_kwargs=None):
        super().__init__(model)
        self.trainer = trainer
        self.early_stopping = early_stopping
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
    
    def split_kwargs_into_model_and_trainer(self, kwargs):
        model_kwargs = {key: value for key, value in kwargs.items() if key in self.MODEL_ARCHITECURE_KEYS}
        trainer_kwargs = {key: value for key, value in kwargs.items() if key in self.TRAINER_KEYS}
        
        if "criterion" in trainer_kwargs and "str_loss" not in trainer_kwargs:
            trainer_kwargs["str_loss"] = trainer_kwargs.pop("criterion")
        if "optimizer" in trainer_kwargs and "str_opt" not in trainer_kwargs:
            trainer_kwargs["str_opt"] = trainer_kwargs.pop("optimizer")
        if "learning_rate" in trainer_kwargs and "lr" not in trainer_kwargs:
            trainer_kwargs["lr"] = trainer_kwargs.pop("learning_rate")
        
        return model_kwargs, trainer_kwargs
    
    # Allows user to create new trainer, typically for hyperparameter tuning
    def override_trainer(self, trainer_kwargs):
        raise NotImplementedError
    
    def override_model(self, model_args, model_kwargs) -> None:
        super().override_model(model_args, model_kwargs)
        self.override_trainer()
    
    def reset_model(self) -> None:
        self.override_model(self.model_args, self.model_kwargs)
    
    # TODO: Move predict to Classifier class?? Or bump up to BaseTorchModel? Or Keep here? 
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

    # TODO: Move get_accuracy and get_auc to Evaluation class??
    # def get_accuracy(
    #     self,
    #     data: npt.NDArray,
    #     label: npt.NDArray,
    # ):
    #     # obtain the prediction
    #     y_pred = np.argmax(self.predict(data), axis=1)
    #     y_pred = self._check_input(y_pred)

    #     # if label is one-hot encoded, convert it to integer
    #     y_real = self._check_input(label)

    #     return np.sum(y_pred == y_real) / y_real.shape[0]
    
    # def get_auc(
    #     self,
    #     scores: npt.NDArray,
    #     label: npt.NDArray,
    # ) -> npt.NDArray:
    #     auc = torchmetrics.AUROC(task="multiclass", num_classes=self.model.n_class)(
    #         torch.Tensor(scores), torch.Tensor(label).to(torch.long)
    #     )

    #     return ptu.to_numpy(auc)
    

def init_model_torch(
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
        # initialize the model
        model = RNNModelWrapper(
            *model_args,
            **model_kwargs,
        )
    else:
        raise NotImplementedError

    return model
