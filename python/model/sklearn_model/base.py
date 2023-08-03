from types import MappingProxyType

import numpy as np
from sklearn.dummy import DummyClassifier

from ..base import BaseModel
from .discriminant_analysis import LDAModel, QDAModel
from .support_vector_machine import SVMModel
from .gaussian_process import GPModel
from .tree_based_model import RandomForestModel
from .boosting_method import AdaBoostModel, GradientBoostingModel


class SKLearnModel(BaseModel):
    def __init__(self):
        super().__init__()

        # initiate the fields
        self.n_class = None
        self.model = DummyClassifier()
        self.bool_torch = False

    def train(self, data, label):
        # train the model
        data = self._check_input(data)
        self.n_class = len(np.unique(label))
        self.model.fit(data, label)

    def predict(self, data):
        # generate the predictions
        data = self._check_input(data)
        return self.model.predict(data)

    def get_accuracy(self, data, label):
        # generate the accuracy
        data = self._check_input(data)
        return self.model.score(data, label)

    def predict_proba(self, data):
        # generate the prediction probabilities
        data = self._check_input(data)

        # organize prediction score based on documentation of sklearn.metrics.roc_auc_score
        if self.n_class == 2:
            scores = self.model.predict_proba(data)[:, 1]  # type: ignore
        else:
            scores = self.model.predict_proba(data)

        return scores


def init_model_sklearn(
    str_model: str,
    model_args: list | tuple = tuple(),
    model_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
    ensemble_args: list | tuple = tuple(),
    ensemble_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
):

    # parse the arguments and initialize the corresponding model
    if str_model == "LDA":
        model = LDAModel(model_args, model_kwargs)
    elif str_model == "QDA":
        model = QDAModel(model_args, model_kwargs)
    elif str_model == "SVM":
        model = SVMModel(model_args, model_kwargs)
    elif str_model == "RF":
        model = RandomForestModel(model_args, model_kwargs)
    elif str_model == "GP":
        model = GPModel(model_args, model_kwargs)
    elif str_model == "AdaBoost":
        model = AdaBoostModel(model_args, model_kwargs, ensemble_args, ensemble_kwargs)
    elif str_model == "GBoost":
        model = GradientBoostingModel(model_args, model_kwargs)
    else:
        raise NotImplementedError("f{str_model} is not implemented yet.")

    return model
