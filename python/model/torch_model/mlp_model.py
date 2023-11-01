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


class TorchMLPClassifier(nn.Module, BaseTorchClassifier):
    def __init__(
            self,
            n_input=64,
            n_class=2,
            act_func=nn.LeakyReLU(),
            n_layer=3,
            hidden_size=32,
            dropout=0.5,
    ):
        super().__init__()

        # initialize the fields
        self.n_input = n_input
        self.n_class = n_class
        self.n_layer = n_layer
        self.act_func = super().get_activation(act_func)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # define n_output based on the loss function
        # if str_loss == "CrossEntropy":
        #     n_output = n_class
        # else:
        #     n_output = 1
        self.n_output = self.n_class

        # initialize the MLP layers
        if n_layer > 0:
            for i in range(n_layer):
                # append the linear layer
                if i == 0:
                    self.layers.append(nn.Linear(self.n_input, self.hidden_size))
                else:
                    self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

                # append the activation function
                self.layers.append(self.act_func)

                # append the dropout
                self.layers.append(nn.Dropout(self.dropout))

            # append the output layer
            self.layers.append(nn.Linear(self.hidden_size, self.n_output))

        else:
            # append the output layer only
            self.layers.append(nn.Linear(self.n_input, self.n_output))

            # append the activation function
            self.layers.append(self.act_func)

            # append the dropout
            self.layers.append(nn.Dropout(self.dropout))
        # sanity check
        assert len(self.layers) == 3 * self.n_layer + 1 if self.n_layer > 0 else 3

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
    

class TorchMLPTrainer(BaseTorchTrainer):
    def __init__(
        self,
        model,
        n_class=2,
        str_loss='CrossEntropy',
        str_reg='L2',
        lam=1e-5,
        str_opt='Adam',
        lr=1e-4,
        transform=None,
        target_transform=None,
        batch_size=32,
        n_epoch=100,
        bool_shuffle=True,
        bool_verbose=False,
        early_stopper=None,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0
    ):
        super().__init__(model,
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
            early_stopper,
            bool_verbose,
            bool_use_ray,
            bool_use_gpu,
            n_gpu_per_process
        )
    
    
    # def train(
    #     self,
    #     train_data,
    #     train_label,
    #     valid_data: npt.NDArray | None = None,
    #     valid_label: npt.NDArray | None = None,
    #     batch_size: int = 32,
    #     n_epoch: int = 100,
    #     bool_early_stopping: bool = True,
    #     es_patience: int = 5,
    #     es_delta: float = 1e-5,
    #     es_metric: str = "val_loss",
    #     bool_shuffle: bool = True,
    #     bool_verbose: bool = True,
    # ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
    #     # ensure that data is in the right format
    #     if train_data.ndim == 3:
    #         raise ValueError("Need to convert to 2D data first")

    #     vec_avg_loss, vec_avg_valid_loss = self.trainer.train(
    #         train_data,
    #         train_label,
    #         valid_data,
    #         valid_label,
    #         batch_size,
    #         n_epoch,
    #         bool_early_stopping,
    #         es_patience,
    #         es_delta,
    #         es_metric,
    #         bool_shuffle,
    #         bool_verbose,
    #     )

    #     return vec_avg_loss, vec_avg_valid_loss

# @ray.remote(num_gpus=0.125)
class TorchMLPModel(BaseTorchModel):
    def __init__(
        self,
        n_input=64,
        n_class=2,
        n_layer=3,
        hidden_size=32,
        dropout: float = 0.5,
        lam=1e-5,
        act_func="relu",
        str_reg="L2",
        str_loss="CrossEntropy",
        str_opt="Adam",
        lr=1e-4,
        early_stopper=None,
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
        self.model = TorchMLPClassifier(
            **self.model_kwargs
        )
        self.model.to(self.device)
        
        self.trainer = TorchMLPTrainer(
            self.model, **self.trainer_kwargs
        )

    # def predict(self, data: npt.NDArray) -> npt.NDArray:
    #     # ensure that data is in the right format
    #     if data.ndim == 3:
    #         raise ValueError("Need to convert to 2D data first")

    #     return super().predict(data)
    
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
    
    def get_auc(
        self,
        scores: npt.NDArray,
        label: npt.NDArray,
    ) -> npt.NDArray:
        auc = torchmetrics.AUROC(task="multiclass", num_classes=self.model.n_class)(
            torch.Tensor(scores), torch.Tensor(label).to(torch.long)
        )

        return ptu.to_numpy(auc)
    
    def override_model(self, kwargs: dict) -> None:
        model_kwargs, trainer_kwargs = self.split_kwargs_into_model_and_trainer(kwargs)
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.model = TorchMLPClassifier(**model_kwargs)
        self.trainer = TorchMLPTrainer(self.model, **trainer_kwargs)
        self.model.to(self.device)
    
    def reset_model(self) -> None:
        #self.override_model(self.model_kwargs | self.trainer_kwargs)
        self.model = TorchMLPClassifier(**self.model_kwargs)
        self.trainer = TorchMLPTrainer(self.model, **self.trainer_kwargs)
        self.model.to(self.device)
        


@ray.remote(num_cpus=1, num_gpus=0.2)
class MLPModelWrapperRay(TorchMLPModel):
    def __init__(self, *args, **kwargs):
        # TODO: actor is deprecated, need to remove
        super().__init__(*args, **kwargs)


        
    # def update_training_params(self, epochs, batch_size, criterion='CrossEntropy', optimizer='adam', lr=0.1):
    #     self.epochs = epochs
    #     self.batch_size = batch_size
    #     self.criterion = self.create_criterion(criterion)
    #     self.optimizer = self.create_optimizer(optimizer, lr)
    #     # TODO: Might be best to not instantiate optimizer, and pass keywords into NeuralNetClassifier with 'optimizer__' prefix
    #     return {
    #         "max_epochs": self.epochs,
    #         "criterion": self.criterion,
    #         "optimizer": self.optimizer,
    #         "batch_size": self.batch_size,
    #         "callbacks": self.callbacks
    #     }
# class SKCompatWrapper():
#     def __init__(self, model, criterion='CrossEntropy', optimizer='adam', lr=0.1, epochs=10, batch_size=32):
#         self.model = model
#         self.criterion = self.create_criterion(criterion)
#         self.optimizer = self.create_optimizer(optimizer, lr)
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.losses = []
    
#     def create_optimizer(self, optimizer, lr):
#         if isinstance(optimizer, optim.Optimizer):
#             return optimizer
#         elif optimizer == 'adam':
#             return optim.Adam(self.model.parameters(), lr=lr)
#         elif optimizer == 'sgd':
#             return optim.SGD(self.model.parameters(), lr=lr)
#         else:
#             raise ValueError(f'Optimizer {optimizer} not supported')
    
#     def create_criterion(self, criterion):
#         if isinstance(criterion, nn.Module):
#             return criterion
#         elif criterion == 'CrossEntropy':
#             return nn.CrossEntropyLoss()
#         elif criterion == 'mse':
#             return nn.MSELoss()
#         else:
#             raise ValueError(f'Criterion {criterion} not supported')
    
#     def update_training_params(self, epochs, batch_size, criterion='CrossEntropy', optimizer='adam', lr=0.1):
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.criterion = self.create_criterion(criterion)
#         self.optimizer = self.create_optimizer(optimizer, lr)

# TODO: Find home for this code. "fit" call should probably go in torch wrappers (seems Jiaang called it 'train')?
#     def fit(self, X, y):
#         y = torch.tensor(OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray(), dtype=torch.float32)
#         dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), y)
#         data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
#         for epoch in range(self.epochs):
#             for inputs, labels in data_loader:
#                 #labels = torch.tensor(OneHotEncoder(categories=[0,1,2,3,4]).fit_transform(labels.reshape(-1, 1)).toarray(), dtype=torch.float32)
#                 #labels = nn.functional.one_hot(labels, num_classes=5)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs.view_as(labels), labels)
#                 loss.backward()
#                 wandb.log({'loss': loss.item()})
#                 self.optimizer.step()
#         return self

#     def predict(self, X):
#         with torch.no_grad():
#             inputs = torch.tensor(X, dtype=torch.float32)
#             outputs = self.model(inputs)
#         #return outputs.numpy()
#         # argmax converts it back to muliclass labels for sklearn
#         return np.argmax(outputs.numpy(), axis=-1)
    
#     def predict_proba(self, X):
#         with torch.no_grad():
#             inputs = torch.tensor(X, dtype=torch.float32)
#             outputs = self.model(inputs)
#             return torch.softmax(outputs.numpy())

#     def score(self, X, y):
#         y_pred = self.predict(X)
#         return accuracy_score(y, y_pred)

#     def get_params(self, deep=True):
#         return {
#             "model": self.model,
#             "criterion": self.criterion,
#             "optimizer": self.optimizer,
#             "epochs": self.epochs,
#             "batch_size": self.batch_size
#         }

# class SKCompatWrapperExternal(BaseModel):
    
#     def __init__(self, model, X, y, validation, scoring):
#         super().__init__(model, X, y, validation, scoring)
    
    
#     def override_model(self, model_args, model_kwargs) -> None:
#         config = model_kwargs
#         architecture_config = [config['n_input'], config['n_class'], config['activation'], config['criterion'], config['n_layer'], config['hidden_size'], config['dropout']]
#         self.model = SKCompatWrapper(TorchMLPModel(*architecture_config))
#         #self.model.to(self.device)
#         self.model.update_training_params(config['epochs'], config['batch_size'], config['criterion'], config['optimizer'], config['lr'])
        
        
#     def wandb_train(self, config=None):
#         X_train, y_train = self.fetch_data()
#         # Initialize a new wandb run
#         with wandb.init(config=config, dir=self.output_dir, group=self.wandb_group, tags=self.wandb_tags):
#             # If called by wandb.agent this config will be set by Sweep Controller
#             config = wandb.config

#             self.override_model((), config)
            
#             # Create condition to check if kfold, groupedkfold, stratifiedkfold, or simple train test split

#             self.model.fit(X_train, np.argmax(y_train, axis=-1))
#             wandb.log({'accuracy': self.model.score(X_train, np.argmax(y_train, axis=-1))})
#             # # Evaluate predictions
#             # results = evaluate_model(self.model, X_train, np.argmax(y_train, axis=-1), self.validation, self.scoring)            
            
#             # TODO: Figure out why cross-validation code below yields horrible accuracy results...
#             # #Drop prefixes for logging
#             # mean_results = {f'{k.split("_", 1)[-1]}_mean': np.mean(v) for k, v in results.items()}
#             # std_results = {f'{k.split("_", 1)[-1]}_std': np.std(v) for k, v in results.items()}

#             # # Log model performance metrics to W&B
#             # wandb.log(std_results)
#             # wandb.log(mean_results)
    
