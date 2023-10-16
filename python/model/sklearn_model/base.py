import numpy as np
from sklearn.dummy import DummyClassifier

from ..base import BaseModel


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
