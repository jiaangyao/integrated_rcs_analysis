# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# from catboost import CatBoostClassifier
from model.base import BaseModel

import numpy as np


class XGBoostModel:
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        # self.model = XGBClassifier(*model_args, **model_kwargs)


class LightXGBoostModel(BaseModel):
    def __init__(self, model_kwargs={}):
        self.model_kwargs = model_kwargs
        super().__init__(LGBMClassifier(**model_kwargs))

    def override_model(self, model_kwargs):
        self.model_kwargs = model_kwargs
        self.model = LGBMClassifier(**model_kwargs)
    
    def reset_model(self):
        self.model = LGBMClassifier(**self.model_kwargs)


class CatBoostModel:
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        # self.model = CatBoostClassifier(*model_args, **model_kwargs)
