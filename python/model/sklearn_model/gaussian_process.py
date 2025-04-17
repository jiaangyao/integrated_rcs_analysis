from sklearn.gaussian_process import GaussianProcessClassifier

from .base import SKLearnModel


class GPModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = GaussianProcessClassifier(*model_args, **model_kwargs)
