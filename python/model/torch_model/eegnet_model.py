import torch
import torch.nn as nn

import typing as tp

import numpy.typing as npt
import torch.nn as nn
from .torch_utils import init_gpu
import torch
import model.torch_model.torch_utils as ptu
import numpy.typing as npt
import model.torch_model.torch_utils as ptu


from .base import BaseTorchTrainer, BaseTorchModel, BaseTorchClassifier
from .mlp_model import TorchMLPTrainer


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    r'''
    A compact convolutional neural network (EEGNet). For more details, please refer to the following information.

    - Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    - URL: https://arxiv.org/abs/1611.08024
    - Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.Compose([
                        transforms.To2d()
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = EEGNet(chunk_size=128,
                    num_electrodes=32,
                    dropout=0.5,
                    kernel_1=64,
                    kernel_2=16,
                    F1=8,
                    F2=16,
                    D=2,
                    num_classes=2)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (default: :obj:`64`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    '''
    def __init__(self,
                chunk_size: int = 151,
                num_electrodes: int = 60, # Number of recording streams
                in_channels: int = 1, # If raw time series is used, in_channels = 1. Otherwise, number of decompositions (e.g. wavelet filtered channels)
                F1: int = 8,
                F2: int = 16,
                D: int = 2,
                n_class: int = 2,
                kernel_1: int = 64,
                kernel_2: int = 16,
                linear_input_size: int = 128,
                dropout: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.n_class = n_class
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        # Temporal convolution block
        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), groups=1, bias=False),
            #nn.Conv2d(1, self.F1, kernel_size=(1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), groups=1, bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                self.F1 * self.D, (self.num_electrodes, 1),
                                max_norm=1,
                                stride=1,
                                padding=(0, 0),
                                groups=self.F1,
                                bias=False), 
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        # Spatial convolution block
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                    self.F1 * self.D, (1, self.kernel_2),
                    stride=1,
                    padding=(0, self.kernel_2 // 2),
                    bias=False,
                    groups=self.F1 * self.D))
        
        # Separable convolution block
        self.block3 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(linear_input_size, self.n_class, bias=False)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, self.num_electrodes, self.chunk_size)
            # mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        # TODO: Fix this hacky solution for dimensionality...
        if len(x.shape) == 3:
            x = x.unsqueeze(2)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x

class EEGNetTrainer(BaseTorchTrainer):
    def __init__(
        self,
        model,
        early_stopping=None,
        n_class=2,
        str_loss='CrossEntropy',
        str_reg='L2',
        lam=1e-5,
        str_opt='Adam',
        lr=1e-4,
        transform=None,
        target_transform=None,
        batch_size=32,
        n_epoch=20,
        bool_shuffle=True,
        bool_verbose=False,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0
    ):
        super().__init__(
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
            batch_size,
            n_epoch,
            bool_shuffle,
            bool_verbose,
            bool_use_ray,
            bool_use_gpu,
            n_gpu_per_process
        )

class EEGNetModel(BaseTorchModel):
    MODEL_ARCHITECURE_KEYS = [
        "chunk_size",
        "num_electrodes", # Number of recording streams
        "in_channels", # If raw time series is used, in_channels = 1. Otherwise, number of decompositions (e.g. wavelet filtered channels)
        "F1",
        "F2",
        "D",
        "n_class",
        "kernel_1",
        "kernel_2",
        "linear_input_size",
        "dropout"
    ]
    
    def __init__(
        self,
        chunk_size: int = 151,
        num_electrodes: int = 60, # Number of recording streams
        in_channels: int = 1, # If raw time series is used, in_channels = 1. Otherwise, number of decompositions (e.g. wavelet filtered channels)
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_1: int = 64,
        kernel_2: int = 16,
        linear_input_size: int = 128,
        n_class=2,
        dropout: float = 0.5,
        lam=1e-5,
        n_epoch=20,
        batch_size=32,
        act_func="relu",
        str_reg="L2",
        str_loss="CrossEntropy",
        str_opt="Adam",
        lr=1e-4,
        early_stopping=None,
        bool_verbose=False,
        transform=None,
        target_transform=None,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
    ):
        
        self.model_kwargs, self.trainer_kwargs = self.split_kwargs_into_model_and_trainer(locals())
        # TODO: Improve device logic...
        self.device = init_gpu(use_gpu=bool_use_gpu)
        
        # initialize the model
        self.model = EEGNet(
            **self.model_kwargs
        )
        self.model.to(self.device)
        
        # Initialize early stopping
        self.early_stopping = self.get_early_stopper(early_stopping)
        
        # Initialize Trainer
        self.trainer = EEGNetTrainer(
            self.model, self.early_stopping, **self.trainer_kwargs
        )
        super().__init__(self.model, self.trainer, self.early_stopping, self.model_kwargs, self.trainer_kwargs)
    
    
    def override_model(self, kwargs: dict) -> None:
        model_kwargs, trainer_kwargs = self.split_kwargs_into_model_and_trainer(kwargs)
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.model = EEGNet(**model_kwargs)
        self.model.to(self.device)
        if self.early_stopping is not None: self.early_stopping.reset()
        self.trainer = EEGNetTrainer(self.model, self.early_stopping, **trainer_kwargs)
        self.model.to(self.device)
    
    def reset_model(self) -> None:
        #self.override_model(self.model_kwargs | self.trainer_kwargs)
        self.model = EEGNet(**self.model_kwargs)
        self.model.to(self.device)
        if self.early_stopping is not None: self.early_stopping.reset()
        self.trainer = EEGNetTrainer(self.model, self.early_stopping, **self.trainer_kwargs)
        self.model.to(self.device)