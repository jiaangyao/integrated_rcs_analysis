from __future__ import print_function
import typing
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)


def get_model(
    str_model,
    model_params: MappingProxyType | dict = MappingProxyType(dict()),
    adaboost_params: MappingProxyType | dict = MappingProxyType(dict()),
):
    if str_model == "LDA":
        model = LDAModel(model_params)
    elif str_model == "QDA":
        model = QDAModel(model_params)
    elif str_model == "SVM":
        model = SVMModel(model_params)
    elif str_model == "RF":
        model = RandomForestModel(model_params)
    elif str_model == "AdaBoost":
        model = AdaBoostModel(model_params, adaboost_params)
    elif str_model == "GP":
        model = GPModel(model_params)
    elif str_model == "GB":
        model = GradientBoostingModel(model_params)
    else:
        raise NotImplementedError

    return model


class BaseModel:
    def __init__(self):
        self.model = None

    def train(self, data, label):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    @staticmethod
    def _check_input(data: npt.NDArray):
        if len(data.shape) == 1:
            data = data[:, None]
        return data


class SKLearnModel(BaseModel):
    def __init__(self):
        super().__init__()

        # initiate the fields
        self.n_class = None
        self.model = DummyClassifier()

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
        if self.n_class == 2:
            scores = self.model.predict_proba(data)[:, 1]  # type: ignore
        else:
            scores = self.model.predict_proba(data)

        return scores


class LDAModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = LinearDiscriminantAnalysis(**model_params)


class QDAModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = QuadraticDiscriminantAnalysis(**model_params)


class SVMModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = SVC(**model_params)


class GPModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = GaussianProcessClassifier(1.0 * RBF(1.0), **model_params)


class GradientBoostingModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = GradientBoostingClassifier(**model_params)


class AdaBoostModel(SKLearnModel):
    def __init__(self, svm_model_params, adaboost_model_params):
        super().__init__()

        # initialize the model
        self.model = AdaBoostClassifier(
            estimator=SVC(**svm_model_params), **adaboost_model_params
        )


class RandomForestModel(SKLearnModel):
    def __init__(self, model_params):
        super().__init__()

        # initialize the model
        self.model = RandomForestClassifier(**model_params)
