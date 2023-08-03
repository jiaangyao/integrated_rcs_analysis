from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from .base import SKLearnModel


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