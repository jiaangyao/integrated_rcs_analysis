import torch
import torch.nn as nn
from .base import BaseTorchModel, BaseTorchTrainer
from .torch_utils import init_gpu

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, n_class: int = 5, n_channels: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetTrainer(BaseTorchTrainer):
    def __init__(
        self,
        model,
        early_stopping=None,
        n_class=2,
        str_loss="CrossEntropy",
        str_reg="L2",
        lam=1e-5,
        str_opt="Adam",
        lr=1e-4,
        transform=None,
        target_transform=None,
        batch_size=32,
        n_epoch=20,
        bool_shuffle=True,
        bool_verbose=False,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
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
            n_gpu_per_process,
        )


class AlexNetModel(BaseTorchModel):
    MODEL_ARCHITECURE_KEYS = [
        "n_class",
        "n_channels",
        "dropout",
    ]

    def __init__(
        self,
        n_class=2,
        n_channels=1,
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

        (
            self.model_kwargs,
            self.trainer_kwargs,
        ) = self.split_kwargs_into_model_and_trainer(locals())

        self.device = init_gpu(use_gpu=bool_use_gpu)

        # initialize the model
        self.model = AlexNet(**self.model_kwargs)
        self.model.to(self.device)

        # Initialize early stopping
        self.early_stopping = self.get_early_stopper(early_stopping)

        # Initialize Trainer
        self.trainer = AlexNetTrainer(
            self.model, self.early_stopping, **self.trainer_kwargs
        )
        super().__init__(
            self.model,
            self.trainer,
            self.early_stopping,
            self.model_kwargs,
            self.trainer_kwargs,
        )

    def override_model(self, kwargs: dict) -> None:
        model_kwargs, trainer_kwargs = self.split_kwargs_into_model_and_trainer(kwargs)
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.model = AlexNet(**model_kwargs)
        self.model.to(self.device)
        if self.early_stopping is not None:
            self.early_stopping.reset()
        self.trainer = AlexNetTrainer(self.model, self.early_stopping, **trainer_kwargs)
        self.model.to(self.device)

    def reset_model(self) -> None:
        # self.override_model(self.model_kwargs | self.trainer_kwargs)
        self.model = AlexNet(**self.model_kwargs)
        self.model.to(self.device)
        if self.early_stopping is not None:
            self.early_stopping.reset()
        self.trainer = AlexNetTrainer(
            self.model, self.early_stopping, **self.trainer_kwargs
        )
        self.model.to(self.device)
