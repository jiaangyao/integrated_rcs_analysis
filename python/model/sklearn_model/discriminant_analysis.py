from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

# from .base import SKLearnModel
from model.base import BaseModel


class LDAModel(BaseModel):        
    def __init__(self, model_kwargs={}):
        self.model_kwargs = model_kwargs
        super().__init__(LinearDiscriminantAnalysis(**model_kwargs))

    def override_model(self, model_kwargs):
        self.model_kwargs = model_kwargs
        self.model = LinearDiscriminantAnalysis(**model_kwargs)

    def reset_model(self):
        self.model = LinearDiscriminantAnalysis(**self.model_kwargs)


class QDAModel(BaseModel):
    # def __init__(self, model_args, model_kwargs):
    #     super().__init__()

    #     # initialize the model
    #     self.model = QuadraticDiscriminantAnalysis(*model_args, **model_kwargs)
        
    def __init__(self, model_kwargs={}):
        self.model_kwargs = model_kwargs
        super().__init__(QuadraticDiscriminantAnalysis(**model_kwargs))

    def override_model(self, model_kwargs):
        self.model_kwargs = model_kwargs
        self.model = QuadraticDiscriminantAnalysis(**model_kwargs)

    def reset_model(self):
        self.model = QuadraticDiscriminantAnalysis(**self.model_kwargs)

