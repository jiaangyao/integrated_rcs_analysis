import numpy as np
import numpy.typing as npt

from pydantic import BaseModel as DanticBaseModel

from sklearn.model_selection import train_test_split


# MLdata is a (of type) dataclass that holds training and testing data for machine learning models.
# For info on dataclasses see: https://realpython.com/python-data-classes/
class MLData(DanticBaseModel):
    """
    A class to hold training and testing data for machine learning models.
    All data fields are expected to be numpy ndarrays.
    Using Pydantic for data validation.
    """

    X: npt.NDArray | None = None
    y: npt.NDArray | None = None
    groups: npt.NDArray | list | tuple | None = None
    X_train: npt.NDArray | None = None
    y_train: npt.NDArray | None = None
    groups_train: npt.NDArray | list | tuple | None = None
    one_hot_encoded: bool = False

    X_test: npt.NDArray | None = None
    y_test: npt.NDArray | None = None
    groups_test: npt.NDArray | list | tuple | None = None

    # Should be a tuple of the form (train_inds, val_inds, test_inds), where each of the three elements is a list of indices or empty list/array.
    train_val_test_indicies: tuple | None = None

    # Ideally, folds is a list of dictionaries, where each dictionary contains the indices for the training and validations sets,
    # under keys 'train' and 'val', respectively.
    # NOTE: FOLDS SHOULD BE CREATED USING THE TRAINING DATA (i.e. X_train) ONLY
    folds: npt.NDArray | list | tuple | None = None
    
    # Setup default for get_fold_by_index to True, so that the default behavior is to get the fold by index of training data.
    # If get_fold_by_index is False, then each fold saves the training and validation sets explicitly as a tuple of the form (X_train, y_train, X_val, y_val).
    # This is useful if the training data needs to be class imbalance corrected or augmented, and the folds need to be updated. 
    # Running class imbalance correction or data augmentation on the training data will turn this flag to False.
    get_fold_by_index_bool: bool = True
    metadata: dict | None = None

    def __init__(self, **data):
        super().__init__(**data)

        print(
            "Assigning X_train and y_train to X and y, respectively... until train-test split is called."
        )
        if self.X_train is None:
            self.X_train = self.X
        if self.y_train is None:
            self.y_train = self.y

    # This is to allow numpy arrays to be passed in as dataclass fields
    class Config:
        arbitrary_types_allowed = True

    def train_test_split(self, train_inds, test_inds):
        """
        Splits the data into training and testing sets based on the indices provided.

        Args:
            train_inds (list): A list of indices to use for the training set.
            test_inds (list): A list of indices to use for the testing set.

        Returns:
            None
        """
        self.X_train = self.X[train_inds]
        self.y_train = self.y[train_inds]
        self.X_test = self.X[test_inds]
        self.y_test = self.y[test_inds]
        if self.groups is not None:
            self.groups_train = self.groups[train_inds]
            self.groups_test = self.groups[test_inds]

    # TODO: Define the following 'get_...' funcs as @property methods?
    def get_training_data(self):
        return self.X_train, self.y_train

    def get_testing_data(self):
        return self.X_test, self.y_test
    
    def get_fold(self, fold_num):
        """
        Returns the training and testing data for the fold specified by fold_ind.
        """
        if self.get_fold_by_index_bool:
            return self.get_fold_by_index(fold_num)
        else:
            return self.get_fold_by_explicit(fold_num)
    
    def get_training_folds(self):
        """
        Returns the training data for all folds.
        """
        if self.get_training_folds_by_index:
            return self.get_training_folds_by_index()
        else:
            return self.get_training_folds_by_explicit()
    
    
    def get_validation_folds(self):
        """
        Returns the training data for all folds.
        """
        if self.get_training_folds_by_index:
            return self.get_validation_folds_by_index()
        else:
            return self.get_validation_folds_by_explicit()


    def get_fold_by_index(self, fold_num):
        """
        Returns the training and testing data for the fold specified by fold_ind.
        """
        train_inds = self.folds[fold_num]["train"]
        val_inds = self.folds[fold_num]["val"]
        return (
            self.X_train[train_inds],
            self.y_train[train_inds],
            self.X_train[val_inds],
            self.y_train[val_inds],
        )


    def get_fold_by_explicit(self, fold_num):
        """
        Returns the training and testing data for the fold number specified by fold_ind.
        """
        return self.folds[fold_num]
    


    def get_training_folds_by_index(self):
        """
        Returns the training data for all folds.
        """
        return [
            (self.X_train[fold["train"]], self.y_train[fold["train"]])
            for fold in self.folds
        ]
    
    def get_training_folds_by_explicit(self):
        """
        Returns the training data for all folds.
        """
        return [
            (fold[0], fold[1])
            for fold in self.folds
        ]


    def get_validation_folds_by_index(self):
        """
        Returns the testing data for all folds.
        """
        return [
            (self.X_train[fold["val"]], self.y_train[fold["val"]])
            for fold in self.folds
        ]
    
    def get_validation_folds_by_explicit(self):
        """
        Returns the training data for all folds.
        """
        return [
            (fold[2], fold[3])
            for fold in self.folds
        ]



    def assign_train_val_test_indices(self, train_inds=[], val_inds=[], test_inds=[]):
        """
        Assigns the training, validation, and testing indicies to the dataclass fields.
        """
        self.train_val_test_indicies = (train_inds, val_inds, test_inds)
    
    
    def override_folds(self, folds: list):
        """
        If training data needs to be class imbalance corrected or augmented, the folds need to be updated. This method re
        """
        assert len(folds) == len(self.folds), "The number of folds provided does not match the number of folds in the data."
        self.folds = folds
        self.get_fold_by_index_bool = False