from typing import Optional

import numpy.typing as npt

from pydantic import BaseModel as DanticBaseModel


# MLdata is a type=dataclass that holds training and testing data for machine learning models.
# For info on dataclasses see: https://realpython.com/python-data-classes/
class MLData(DanticBaseModel):
    """
    A class to hold training and testing data for machine learning models.
    All data fields are expected to be numpy ndarrays.
    Using Pydantic for data validation.
    """

    # TODO: Clean up and use numpy typing
    X_train: npt.NDArray
    y_train: npt.NDArray
    groups: npt.NDArray | None = None
    X_val: npt.NDArray | None = None
    y_val: npt.NDArray | None = None
    X_test: npt.NDArray | None = None
    y_test: npt.NDArray | None = None
    metadata: dict | None = None

    # This is to allow numpy arrays to be passed in as dataclass fields
    class Config:
        arbitrary_types_allowed = True

    def training_data(self):
        return self.X_train, self.y_train

    def testing_data(self):
        return self.X_test, self.y_test

    def validation_data(self):
        return self.X_val, self.y_val
