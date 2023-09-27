#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from model.base import BaseModel

import wandb
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
import numpy as np


class XGBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        #self.model = XGBClassifier(*model_args, **model_kwargs)


class LightXGBoostModel(BaseModel):
    def __init__(self, X_train, y_train, val_object, scoring, model_kwargs, X_test=None, y_test=None, test_model=False):
        super().__init__(LGBMClassifier(**model_kwargs), X_train, y_train, val_object, scoring,  X_test, y_test)

        # # initialize the model
        # self.model = LGBMClassifier(**model_kwargs)
        
        # self.X_train = X_train
        # self.y_train = y_train
        
        # self.X_test = X_test
        # self.y_test = y_test
        
        # self.validation = val_object
        # self.scoring = scoring
        
        self.test_model = test_model
        
    def override_model(self, model_args, model_kwargs):
        self.model = LGBMClassifier(*model_args, **model_kwargs)


class CatBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        # self.model = CatBoostClassifier(*model_args, **model_kwargs)