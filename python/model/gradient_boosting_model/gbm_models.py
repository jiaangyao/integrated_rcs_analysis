from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


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


class CatBoostModel():
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = CatBoostClassifier(*model_args, **model_kwargs)