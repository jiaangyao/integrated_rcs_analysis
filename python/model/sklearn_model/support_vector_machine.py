from sklearn.svm import SVC

from .base import SKLearnModel


class SVMModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = SVC(*model_args, **model_kwargs)