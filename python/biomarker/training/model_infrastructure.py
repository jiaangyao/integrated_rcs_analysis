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

        # oragnize prediction score based on documention of sklearn.metrics.roc_auc_score
        if self.n_class == 2:
            scores = self.model.predict_proba(data)[:, 1]  # type: ignore
        else:
            scores = self.model.predict_proba(data)

        return scores


class LDAModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = LinearDiscriminantAnalysis(*model_args, **model_kwargs)


class QDAModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = QuadraticDiscriminantAnalysis(*model_args, **model_kwargs)


class SVMModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = SVC(*model_args, **model_kwargs)


class GPModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = GaussianProcessClassifier(*model_args, **model_kwargs)


class GradientBoostingModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = GradientBoostingClassifier(*model_args, **model_kwargs)


class AdaBoostModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs, ensemble_args, ensemble_kwargs):
        super().__init__()

        # initialize the model
        self.model = AdaBoostClassifier(
            *ensemble_args,
            estimator=SVC(*model_args, **model_kwargs),
            **ensemble_kwargs
        )


class RandomForestModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = RandomForestClassifier(*model_args, **model_kwargs)
