import typing as tp

import ray
import numpy.typing as npt
import torch.nn as nn
from .torch_utils import init_gpu
import torchmetrics
import torch
import model.torch_model.torch_utils as ptu
import numpy as np
import numpy.typing as npt
import model.torch_model.torch_utils as ptu
from dataset.torch_dataset import NeuralDataset, NeuralDatasetTest
from torch.utils.data import DataLoader

from .base import BaseTorchTrainer, BaseTorchModel, BaseTorchClassifier

class TorchCNNClassifier(nn.Module, BaseTorchClassifier):
    def __init__(
            self,
            in_channels=1,  # Number of channels in the input image
            img_size=32,    # Image height and width (assuming square images)
            n_class=2,
            act_func=nn.LeakyReLU(),
            n_conv_layers=2,
            n_fc_layers=2,
            hidden_size=32,
            dropout=0.5,
    ):
        super().__init__()

        # Initialize the fields
        self.in_channels = in_channels
        self.img_size = img_size
        self.n_class = n_class
        self.n_conv_layers = n_conv_layers
        self.n_fc_layers = n_fc_layers
        self.act_func = super().get_activation(act_func)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Define the convolutional layers
        for i in range(n_conv_layers):
            in_ch = self.in_channels if i == 0 else hidden_size
            self.conv_layers.append(nn.Conv2d(in_ch, hidden_size, kernel_size=3, stride=1, padding=1))
            self.conv_layers.append(self.act_func)
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.flat_size = (img_size // (2**n_conv_layers))**2 * hidden_size

        # Define fully connected layers
        for i in range(n_fc_layers):
            in_features = self.flat_size if i == 0 else hidden_size
            self.fc_layers.append(nn.Linear(in_features, hidden_size))
            self.fc_layers.append(self.act_func)
            self.fc_layers.append(nn.Dropout(self.dropout))

        # Define the output layer
        self.fc_layers.append(nn.Linear(hidden_size, self.n_class))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.flat_size)  # Flatten the tensor

        for layer in self.fc_layers:
            x = layer(x)

        return x
