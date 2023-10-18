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
from typing import Optional

import numpy.typing as npt

from pydantic import BaseModel as DanticBaseModel

from sklearn.model_selection import train_test_split


# MLdata is a type=dataclass that holds training and testing data for machine learning models.
# For info on dataclasses see: https://realpython.com/python-data-classes/
class MLData(DanticBaseModel):
    """
    A class to hold training and testing data for machine learning models.
    All data fields are expected to be numpy ndarrays.
    Using Pydantic for data validation.
    """

    # TODO: Clean up and use numpy typing
    X: npt.NDArray
    y: npt.NDArray
    X_train: npt.NDArray | None = None
    y_train: npt.NDArray | None = None
    groups: npt.NDArray | list | tuple | None = None
    X_val: npt.NDArray | None = None
    y_val: npt.NDArray | None = None
    X_test: npt.NDArray | None = None
    y_test: npt.NDArray | None = None
    metadata: dict | None = None
    
    if X_train is None:
        X_train = X
    if y_train is None:
        y_train = y

    # This is to allow numpy arrays to be passed in as dataclass fields
    class Config:
        arbitrary_types_allowed = True
    
    def train_test_split(self, test_size=0.2, random_seed=42, shuffle=True):
        """
        Split data into train, and test sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_seed, shuffle=shuffle
        )
        
    
    def train_val_test_split(self, val_size=0.2, test_size=0.2, random_seed=42, shuffle=True):
        """
        Split data into train, validation, and test sets using scikit-learn's train_test_split.

        Parameters:
        - data (array-like): The data to be split.
        - val_ratio (float): The ratio of the validation set. Default is 0.2.
        - test_ratio (float): The ratio of the test set. Default is 0.2.
        - random_seed (int, optional): Random seed for reproducibility. Default is 42.
        """
        
        # Make sure val_ratio and test_ratio are valid
        assert 0 <= val_size < 1, "Validation ratio must be between 0 and 1"
        assert 0 <= test_size < 1, "Test ratio must be between 0 and 1"
        assert val_size + test_size < 1, "Validation and test ratios combined must be less than 1"

        train_size = 1 - val_size - test_size

        # Split data into train and temp (temp will be split into val and test)
        self.X_train, X_tmp, self.y_train, y_tmp = train_test_split(
            self.X, self.Y, test_size=1-train_size, random_state=random_seed, shuffle=shuffle
        )

        # Calculate validation split ratio from remaining data
        val_split = val_size / (val_size + test_size)
        
        # Split temp_data into validation and test data
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_tmp, y_tmp, test_size=1-val_split, random_state=random_seed, shuffle=shuffle
        )


    def get_training_data(self):
        return self.X_train, self.y_train

    def get_testing_data(self):
        return self.X_test, self.y_test

    def get_validation_data(self):
        return self.X_val, self.y_val
