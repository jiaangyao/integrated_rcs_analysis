from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import wandb
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate


class XGBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = XGBClassifier(*model_args, **model_kwargs)


class LightXGBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = LGBMClassifier(*model_args, **model_kwargs)
        
        self.X_train = None
        self.y_train = None
        
        # Place holder for better design choice
        self.cv_model = StratifiedKFold(
            n_splits=parameters["cross_val"],
            random_state=parameters["random_state"],
            shuffle=True,
        )
        
    def override_model(self, model_args, model_kwargs):
        self.model = LGBMClassifier(*model_args, **model_kwargs)
    
    def fetch_data(self):
        return self.X_train, self.y_train
        
    def wandb_train(self):
        X_train, y_train = self.fetch_data()
        # Initialize a new wandb run
        with wandb.init(config=config):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config

            self.model = LGBMClassifier(**config['parameters'])
            
            # Create condition to check if kfold, groupedkfold, stratifiedkfold, or simple train test split
            
            self.model.fit(X_train, y_train)

            # Predict on test set
            # y_preds = self.model.predict(X_test)
            
            # Evaluate predictions
            cv_results = cross_validate(
                self.model,
                X,
                y,
                cv=cv,
                scoring=score_dict,
                n_jobs=parameters["cross_val"] + 1,
            )

            # Log model performance metrics to W&B
            wandb.log(cv_results)

                


class CatBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = CatBoostClassifier(*model_args, **model_kwargs)