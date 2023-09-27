import numpy.typing as npt
from hydra import compose
from training_eval.model_utils import *
import wandb
import numpy as np

# TODO: move all constants to constants directory
_VEC_MODEL_DYNAMICS_ONLY = ["RNN"]


class BaseModel:
    """Base model class to be inherited by all sklearn and pytorch models
    """    
    def __init__(self, model, X_train, y_train, validation=None, scoring=None, X_test=None, y_test=None,  groups=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.validation = validation
        self.scoring = scoring
        self.groups = groups

    def fetch_data(self):
        return self.X_train, self.y_train
    
    def fetch_test_data(self):
        return self.X_test, self.y_test
    
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
    
    def wandb_train(self, config=None):
        X_train, y_train = self.fetch_data()
        # Initialize a new wandb run
        with wandb.init(config=config, dir=self.output_dir):
            # If called by wandb.agent this config will be set by Sweep Controller
            config = wandb.config

            self.override_model((), config)
            
            # Create condition to check if kfold, groupedkfold, stratifiedkfold, or simple train test split

            # Evaluate predictions
            results = evaluate_model(self.model, X_train, y_train, self.validation, self.scoring)
            #results = cross_validate(self.model, X_train, y_train, cv=self.validation, scoring=self.scoring)
            
            #Drop prefixes for logging
            mean_results = {f'{k.split("_", 1)[-1]}_mean': np.mean(v) for k, v in results.items()}
            std_results = {f'{k.split("_", 1)[-1]}_std': np.std(v) for k, v in results.items()}

            # Log model performance metrics to W&B
            wandb.log(std_results)
            wandb.log(mean_results)
            
            # TODO: Implement if test_model is true, then log results on hold-out test set
            # if self.test_model:
            #     self.model.fit(X_train, y_train)
            #     X_test, y_test = self.fetch_test_data()
            #     y_preds = self.model.predict(X_test)

    
    def train(self, *args, **kwargs) -> None:
        X_train, y_train = self.fetch_data()
        results = cross_validate(self.model, X_train, y_train, cv=self.validation, scoring=self.scoring)
            
        #Drop prefixes for logging
        mean_results = {f'{k.split("_", 1)[-1]}_mean': np.mean(v) for k, v in results.items()}
        std_results = {f'{k.split("_", 1)[-1]}_std': np.std(v) for k, v in results.items()}

        # Log model performance metrics to W&B
        wandb.log(std_results)
        wandb.log(mean_results)

    def predict(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def override_model(self, model_args, model_kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _check_input(data: npt.NDArray):
        if len(data.shape) == 1:
            data = data[:, None]
        return data


class ArbitraryModel(BaseModel):
    """
    This class can be used to wrap any model, then compare in a pipeline with other models
    """
    def __init__(self, model, X_train, y_train, X_test=None, y_test=None, validation=None, scoring=None):
        super().__init__(model, X_train, y_train, X_test, y_test, validation, scoring)


def get_dynamics_model() -> list:
    return _VEC_MODEL_DYNAMICS_ONLY
