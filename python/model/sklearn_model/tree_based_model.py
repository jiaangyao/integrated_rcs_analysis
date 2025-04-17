from sklearn.ensemble import RandomForestClassifier

from .base import SKLearnModel


class RandomForestModel(SKLearnModel):
    def __init__(self, model_args, model_kwargs):
        super().__init__()

        # initialize the model
        self.model = RandomForestClassifier(*model_args, **model_kwargs)
