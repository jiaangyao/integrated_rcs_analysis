from __future__ import print_function
from types import MappingProxyType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ray
import tqdm
import numpy as np
from scipy.special import softmax
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

import utils.torch_utils as ptu
from utils.torch_early_stopping import EarlyStopping
from biomarker.training.torch_dataset import NeuralDataset, NeuralDatasetTest


def get_model(str_model, model_params=MappingProxyType(dict()), adaboost_params=MappingProxyType(dict())):
    if str_model == 'LDA':
        model = LDAModel(model_params)
    elif str_model == 'QDA':
        model = QDAModel(model_params)
    elif str_model == 'SVM':
        model = SVMModel(model_params)
    elif str_model == 'RF':
        model = RandomForestModel(model_params)
    elif str_model == 'AdaBoost':
        model = AdaBoostModel(model_params, adaboost_params)
    elif str_model == 'GP':
        model = GPModel(model_params)
    elif str_model == 'GB':
        model = GradientBoostingModel(model_params)
    else:
        raise NotImplementedError

    return model


def get_model_ray(str_model, model_params=MappingProxyType(dict()), gpu_per_model=0.25):
    n_models = np.floor(1 / gpu_per_model)
    if str_model == 'MLP':
        model = MLPModelWrapper(**model_params)
        # model = [ray.remote(num_gpus=gpu_per_model)(MLPModelWrapper(**model_params)) for _ in range(n_models)]
    else:
        raise NotImplementedError

    return model


def _check_input(data):
    if len(data.shape) == 1:
        data = data[:, None]
    return data


class BaseModel:
    def __init__(self):
        self.model = None

    def train(self, data, label):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError


class SKLearnModel(BaseModel):
    def __init__(self):
        super().__init__()

        # initiate the fields
        self.n_class = None

    def train(self, data, label):
        # train the model
        data = _check_input(data)
        self.n_class = len(np.unique(label))
        self.model.fit(data, label)

    def predict(self, data):
        # generate the predictions
        data = _check_input(data)
        return self.model.predict(data)

    def get_accuracy(self, data, label):
        # generate the accuracy
        data = _check_input(data)
        return self.model.score(data, label)

    def predict_proba(self, data):
        # generate the prediction probabilities
        data = _check_input(data)
        if self.n_class == 2:
            scores = self.model.predict_proba(data)[:, 1]
        else:
            scores = self.model.predict_proba(data)

        return scores


class LDAModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = LinearDiscriminantAnalysis(**model_params)


class QDAModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = QuadraticDiscriminantAnalysis(**model_params)


class SVMModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = SVC(**model_params)


class GPModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = GaussianProcessClassifier(1.0 * RBF(1.0), **model_params)


class GradientBoostingModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = GradientBoostingClassifier(**model_params)


class AdaBoostModel(SKLearnModel):
    def __init__(self, svm_model_params, adaboost_model_params):
        super().__init__()

        # initialize the model
        self.model = AdaBoostClassifier(estimator=SVC(**svm_model_params), **adaboost_model_params)


class RandomForestModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = RandomForestClassifier(**model_params)


class PyTorchModelWrapper(BaseModel):
    def __init__(self, n_class, str_loss, str_act, str_reg, lam, str_opt, lr, transform, target_transform):
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

    def train(self, train_data, train_label, valid_data=None, valid_label=None, batch_size=32, n_epoch=100,
              bool_early_stopping=True, es_patience=5, es_delta=1e-5, bool_shuffle=True, bool_verbose=True):
        # initialize dataset and dataloader for training set
        train_dataset = NeuralDataset(train_data, train_label, transform=self.transform,
                                      target_transform=self.target_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=bool_shuffle)

        # initialize the dataset and dataloader for validation set
        bool_run_validation = valid_data is not None and valid_label is not None
        valid_dataloader = None
        early_stopper = None
        if bool_run_validation:
            valid_dataset = NeuralDataset(valid_data, valid_label)
            valid_dataloader = DataLoader(valid_dataset, batch_size=valid_label.shape[0], shuffle=False)
            early_stopper = EarlyStopping(patience=es_patience, delta=es_delta, verbose=bool_verbose,
                                          bool_save_checkpoint=False)

        # housekeeping and declare optimizer
        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()

        # train the model
        vec_avg_loss = []
        vec_avg_valid_loss = []
        for epoch in range(n_epoch):
            # initialize the loss and the model
            vec_loss = []
            vec_valid_loss = []
            self.model.train()

            # iterate through the dataset
            for idx_batch, (x, y) in enumerate(train_dataloader):
                assert self.model.training, 'make sure your network is in train mode with `.train()`'

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
                self.model.eval()
                with torch.no_grad():
                    for _, (x_valid, y_valid) in enumerate(valid_dataloader):
                        assert not self.model.training, 'make sure your network is in eval mode with `.eval()`'

                        # forward pass
                        y_valid_pred = self.model(x_valid)

                        # compute the loss
                        valid_loss = loss_fn(y_valid_pred, y_valid)
                        vec_valid_loss.append(valid_loss.item())

                    # obtain the average loss
                    valid_loss = np.mean(np.stack(vec_valid_loss, axis=0))
                    vec_avg_valid_loss.append(valid_loss)
                    str_valid_loss = ', Valid Loss: {:.4f}'.format(valid_loss)

            # print the loss
            if bool_verbose:
                print('Epoch: {}, Loss: {:.4f}{}'.format(epoch + 1, loss, str_valid_loss))

            # call early stopping
            if bool_run_validation and bool_early_stopping:
                early_stopper(valid_loss, self.model)
                if early_stopper.early_stop:
                    if bool_verbose:
                        print('Early stopping')
                    break

        # load the last checkpoint with the best model
        self.model = early_stopper.load_model(self.model)
        vec_avg_loss = np.stack(vec_avg_loss, axis=0)
        vec_avg_valid_loss = np.stack(vec_avg_valid_loss, axis=0)

        return vec_avg_loss, vec_avg_valid_loss

    def predict(self, data: np.ndarray):
        # initialize the dataset and dataloader for testing set
        test_dataset = NeuralDatasetTest(data)
        test_dataloader = DataLoader(test_dataset, batch_size=data.shape[0], shuffle=False)

        # initialize the loss and set model to eval mode
        self.model.eval()
        vec_y_pred = []
        with torch.no_grad():
            for _, x_test in enumerate(test_dataloader):
                assert not self.model.training, 'make sure your network is in eval mode with `.eval()`'

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
        y_pred = _check_input(y_pred)

        # if label is one-hot encoded, convert it to integer
        y_real = _check_input(label)

        return np.sum(y_pred == y_real) / y_real.shape[0]

    def predict_proba(self, data):
        class_prob = softmax(self.predict(data), axis=1)

        if class_prob.shape[1] == 2:
            class_prob = class_prob[:, 1]

        return class_prob

    def get_loss(self):
        if self.str_loss == 'CrossEntropy':
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        return loss

    def get_activation(self):
        act_hash_map = ptu.get_act_func()
        if self.str_act in act_hash_map.keys():
            act = act_hash_map[self.str_act]
        else:
            raise NotImplementedError

        return act

    def get_regularizer(self):
        if self.str_reg == 'L2':
            reg_params = {'weight_decay': self.lam}
        else:
            raise NotImplementedError

        return reg_params

    def get_optimizer(self):
        # obtain the regularizer
        reg_params = self.get_regularizer()

        # initialize the optimizer
        if self.str_opt == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, **reg_params)
        else:
            raise NotImplementedError

        return optimizer


class TorchMLPModel(nn.Module):
    def __init__(self, n_input, n_class, act_func, loss_func, n_layer=3, hidden_size=32, dropout=0.5):
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
        for i in range(n_layer):
            if i == 0:
                self.layers.append(nn.Linear(n_input, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        # append the output layer
        self.layers.append(nn.Linear(hidden_size, n_class))

    def forward(self, x):
        # forward pass
        for i in range(self.n_layer - 1):
            x = self.layers[i](x)
            x = self.act_func(x)
            x = nn.Dropout(self.dropout)(x)

        # output layer
        out = self.layers[-1](x)

        return out


# @ray.remote(num_gpus=0.25)
class MLPModelWrapper(PyTorchModelWrapper):
    def __init__(self, n_input, n_class, n_layer=3, hidden_size=32, dropout=0.5, lam=1e-5, str_act='relu',
                 str_reg='L2', str_loss='CrossEntropy', str_opt='Adam', lr=1e-4, transform=None, target_transform=None):
        # initialize the base model
        super().__init__(n_class, str_loss, str_act, str_reg, lam, str_opt, lr, transform, target_transform)

        # initialize fields for MLP params
        self.n_input = n_input
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.dropout = dropout

        # initialize the model
        self.model = TorchMLPModel(n_input, n_class, self.get_activation(), self.get_loss(), n_layer,
                                   hidden_size, dropout)
        self.model = self.model.to(ptu.device)

    def train(self, train_data, train_label, valid_data=None, valid_label=None, batch_size=32, n_epoch=100,
              bool_early_stopping=True, es_patience=5, es_delta=1e-5, bool_shuffle=True, bool_verbose=True):
        # ensure that data is in the right format
        if train_data.ndim == 3:
            raise NotImplementedError('Dynamics data not supported yet')

        vec_avg_loss, vec_avg_valid_loss = super().train(train_data, train_label, valid_data, valid_label,
                                                         batch_size, n_epoch, bool_early_stopping,
                                                         es_patience, es_delta, bool_shuffle, bool_verbose)

        return vec_avg_loss, vec_avg_valid_loss

    def predict(self, data: np.ndarray):
        # ensure that data is in the right format
        if data.ndim == 3:
            raise NotImplementedError('Dynamics data not supported yet')

        return super().predict(data)
