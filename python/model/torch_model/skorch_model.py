import torch.nn as nn

from model.base import BaseModel

import torch.nn as nn
import torch.optim as optim

TRAINING_PARAMS = ['epochs', 'batch_size', 'criterion', 'optimizer', 'lr', 'device']

ARCHITECURE_PARAMS = ['n_input', 'n_class', 'activation', 'n_layer', 'hidden_size', 'dropout']

from skorch import NeuralNetClassifier

# Wrapper class to make PyTorch model compatible with sklearn cross_validate via Skorch
class SkorchModel(BaseModel):
    def __init__(self, model, architecture_kwargs, training_kwargs):
        self.inner_model = model
        # TODO: Consider instantiating the model within NeuralNetClassifier (https://skorch.readthedocs.io/en/latest/user/neuralnet.html#most-important-arguments-and-methods)
        # TODO: Parse training_kwargs to make sure keys match NeuralNetClassifier params
        # self.epochs = epochs
        # self.batch_size = batch_size
        # self.criterion = self.create_criterion(criterion)
        # self.optimizer = self.create_optimizer(optimizer, lr)
        # self.callbacks = callbacks
        architecture_kwargs = {f"module__{k}": v for k, v in architecture_kwargs.items()}
        # Skorch prefers unininstantiated model, optimizer, and criterion
        architecture_kwargs['module'] = self.inner_model.type()
        training_kwargs['optimizer'] = self.return_optimizer(training_kwargs['optimizer'])
        training_kwargs['criterion'] = self.return_criterion(training_kwargs['criterion'])
        
        model = NeuralNetClassifier(**architecture_kwargs, **training_kwargs)
        super().__init__(model)
    
    
    def return_optimizer(self, optimizer):
        if isinstance(optimizer, optim.Optimizer):
            return optimizer
        elif optimizer == 'adam':
            return optim.Adam
        elif optimizer == 'sgd':
            return optim.SGD
        else:
            raise ValueError(f'Optimizer {optimizer} not supported')
    
    def return_criterion(self, criterion):
        if isinstance(criterion, nn.Module):
            return criterion
        elif criterion == 'CrossEntropy':
            return nn.CrossEntropyLoss
        elif criterion == 'mse':
            return nn.MSELoss
        else:
            raise ValueError(f'Criterion {criterion} not supported')
        
    
    def create_optimizer(self, optimizer, lr):
        if isinstance(optimizer, optim.Optimizer):
            return optimizer
        elif optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f'Optimizer {optimizer} not supported')
    
    
    def create_criterion(self, criterion):
        if isinstance(criterion, nn.Module):
            return criterion
        elif criterion == 'CrossEntropy':
            return nn.CrossEntropyLoss()
        elif criterion == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f'Criterion {criterion} not supported')

    
    def override_model(self, model_args, model_kwargs) -> None:
        config = model_kwargs
        if 'architecture' in config.keys():
            architecture_kwargs = config['architecture']
        else:
            architecture_kwargs = {k: v for k, v in config.items() if k in ARCHITECURE_PARAMS}
        
        architecture_kwargs = {f"module__{k}": v for k, v in architecture_kwargs.items()}
        architecture_kwargs['module'] = self.inner_model.type()
        
        
        if 'training' in config.keys():
            training_kwargs = config['training']
        else:
            training_kwargs = {k: v for k, v in config.items() if k in TRAINING_PARAMS}
            
        self.model = NeuralNetClassifier(**architecture_kwargs, **training_kwargs)
        #self.model.to(self.device) <- Should be included in training_kwargs