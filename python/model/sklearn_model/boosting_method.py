from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC

from .base import SKLearnModel


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
