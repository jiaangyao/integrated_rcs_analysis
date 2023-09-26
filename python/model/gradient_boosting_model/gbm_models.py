#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier

import wandb
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
import numpy as np


class XGBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        #self.model = XGBClassifier(*model_args, **model_kwargs)


class LightXGBoostModel():
    def __init__(self, X, y, val_object, scoring, model_kwargs, X_test=None, y_test=None, test_model=False):
        super().__init__()

        # initialize the model
        self.model = LGBMClassifier(**model_kwargs)
        
        self.X_train = X
        self.y_train = y
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.validation = val_object
        self.scoring = scoring
        
        self.test_model = test_model
        
    def override_model(self, model_args, model_kwargs):
        self.model = LGBMClassifier(*model_args, **model_kwargs)
    
    def fetch_data(self):
        return self.X_train, self.y_train
    
    def fetch_test_data(self):
        return self.X_test, self.y_test
    
    # TODO: Move to superclass and then all classes can inherit this method
    def wandb_train(self, config=None):
        X_train, y_train = self.fetch_data()
        # Initialize a new wandb run
        with wandb.init(config=config):
            # If called by wandb.agent this config will be set by Sweep Controller
            config = wandb.config

            self.override_model((), config)
            
            # Create condition to check if kfold, groupedkfold, stratifiedkfold, or simple train test split

            # Evaluate predictions
            # TODO: Implement check if KFold/StrafiedKFold or LeaveOneGroupOut, and acquire results accordingly
            results = cross_validate(self.model, X_train, y_train, cv=self.validation, scoring=self.scoring)
            
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
            #     results = evaluate_model(self.validation, y_test, y_preds)
            #     wandb.log(results)

    def train(self):
        X_train, y_train = self.fetch_data()
        results = cross_validate(self.model, X_train, y_train, cv=self.validation, scoring=self.scoring)
            
        #Drop prefixes for logging
        mean_results = {f'{k.split("_", 1)[-1]}_mean': np.mean(v) for k, v in results.items()}
        std_results = {f'{k.split("_", 1)[-1]}_std': np.std(v) for k, v in results.items()}

        # Log model performance metrics to W&B
        wandb.log(std_results)
        wandb.log(mean_results)

class CatBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        # self.model = CatBoostClassifier(*model_args, **model_kwargs)