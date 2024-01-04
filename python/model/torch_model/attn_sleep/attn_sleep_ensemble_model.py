from .model_container.model import AttnSleep
from ..base import BaseTorchModel, BaseTorchTrainer
from scipy.stats import mode
import numpy as np
from ..torch_utils import init_gpu
from joblib import Parallel, delayed
import model.torch_model.torch_utils as ptu
from dataset.torch_dataset import NeuralDataset, NeuralDatasetTest
from torch.utils.data import DataLoader
import torch
import numpy.typing as npt

class EnsembleTrainer():
    def __init__(self, model_ensemble, early_stopping, **kwargs):
        self.trainers = [
            BaseTorchTrainer(model_ensemble[i], early_stopping, **kwargs) for i in range(len(model_ensemble))
        ]
    
    def train(self, X, y, one_hot_encoded=False):
        assert len(self.trainers) == X.shape[1], "Number of models in ensemble does not match number of channels matrices. Ensure channel dimension is dimension 1 (Can use channel_options: group channels)."
        
        # items = Parallel(n_jobs=len(self.trainers))(delayed(self._train_single_trainer)(trainer, np.expand_dims(X[:,i,:], 1), y, one_hot_encoded) 
        #                                         for i, trainer in enumerate(self.trainers))
        # vec_avg_loss = np.stack([items[i][0] for i in range(len(items))], axis=0)
        
        vec_avg_loss = []
        train_scores = []
        vec_avg_valid_loss = []
        valid_scores = []
        
        for i, trainer in enumerate(self.trainers):
            X_channel = np.expand_dims(X[:,i,:], 1)
            items = trainer.train(X_channel, y, one_hot_encoded)
            vec_avg_loss.append(items[0])
            train_scores.append(items[1])
            vec_avg_valid_loss.append(items[2])
            valid_scores.append(items[3])
        
        #return np.mean(vec_avg_loss), np.mean(train_scores), np.mean(vec_avg_valid_loss), np.mean(valid_scores)
        # TODO: Kluge fix for now... need to figure out how to return the scores for each channel
        return np.mean(vec_avg_loss, axis=0), {}, np.array([]), {}
    
    def _train_single_trainer(self, trainer, X, y, one_hot_encoded):
        return trainer.train(X, y, one_hot_encoded)

    # def train_all_trainers(self, X, y, one_hot_encoded):
    #     Parallel(n_jobs=-1)(delayed(self._train_single_trainer)(trainer, X, y, channel, one_hot_encoded) for channel, trainer in enumerate(self.trainers))


class AttnSleepEnsemble():
    
    def __init__(self, num_models, model_kwargs):
        self.model_ensemble = [
            AttnSleep(**model_kwargs) for _ in range(num_models)
        ]
    
    def eval(self):
        [self.model_ensemble[i].eval() for i in range(len(self.model_ensemble))]


class AttnSleepEnsembleModel(BaseTorchModel):
    MODEL_ARCHITECURE_KEYS = ["N",
            "d_model",
            "d_ff = 120",
            "h",
            "dropout",
            "num_classes",
            "afr_reduced_cnn_size",
    ]
        
    def __init__(self,
            num_models = 3, # number of models in ensemble, i.e. number of channels to analyze/classify
            N = 2,  # number of TCE clones
            d_model = 80,  # set to be 100 for SHHS dataset
            d_ff = 120,   # dimension of feed forward
            h = 5,  # number of attention heads
            dropout = 0.1,
            n_class = 5,
            afr_reduced_cnn_size = 30,
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
            n_gpu_per_process: int | float = 0
        ):
        
        self.num_models = num_models
        self.device = init_gpu(use_gpu=bool_use_gpu)
        self.model_kwargs, self.trainer_kwargs = self.split_kwargs_into_model_and_trainer(locals())
        
        # initialize the model
        self.model = AttnSleepEnsemble(num_models, self.model_kwargs)
        
        [self.model.model_ensemble[i].to(self.device) for i in range(self.num_models)]

        self.early_stopping = self.get_early_stopper(early_stopping)

        self.trainer = EnsembleTrainer(
            self.model.model_ensemble, self.early_stopping, **self.trainer_kwargs
        )

        super().__init__(self.model, self.trainer, self.early_stopping, self.model_kwargs, self.trainer_kwargs)
        
    
    def _predict_single_model(self, data: npt.NDArray, model) -> npt.NDArray:
        # initialize the dataset and dataloader for testing set
        test_dataset = NeuralDatasetTest(data)
        test_dataloader = DataLoader(
            test_dataset, batch_size=data.shape[0], shuffle=False
        )

        # initialize the loss and set model to eval mode
        model.eval()
        vec_y_pred = []
        with torch.no_grad():
            for _, x_test in enumerate(test_dataloader):
                assert (
                    not model.training
                ), "make sure your network is in eval mode with `.eval()`"

                # forward pass
                y_test_pred = model(x_test)

                # append the probability
                vec_y_pred.append(y_test_pred)

        # stack the prediction
        y_pred = torch.cat(vec_y_pred, dim=0)

        # return ptu.to_numpy(y_pred)
        return y_pred
    
    
    def predict(self, X):
        
        assert len(self.model.model_ensemble) == X.shape[1], "Number of models in ensemble does not match number of channels. Ensure channel dimension is dimension 1 (Can use channel_options: group channels)."
        # Return the stacked prediction probabilities for each model
        return torch.stack([
            self._predict_single_model(torch.unsqueeze(X[:,i,:], 1), self.model.model_ensemble[i]) for i in range(self.num_models)
        ], dim=0)
    
    
    def predict_proba(self, X):
        return torch.mean(self.predict(X), axis=0)
    
    
    def predict_classes(self, X):
        return torch.mode(self.predict(X), axis=0)[0]
    
    
    def reset_model(self) -> None:
        # initialize the model
        self.model = AttnSleepEnsemble(self.num_models, self.model_kwargs)
        
        [self.model.model_ensemble[i].to(self.device) for i in range(self.num_models)]

        if self.early_stopping is not None: self.early_stopping.reset()

        self.trainer = EnsembleTrainer(
            self.model.model_ensemble, self.early_stopping, **self.trainer_kwargs
        )