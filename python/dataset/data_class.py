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
    X: npt.NDArray
    y: npt.NDArray
    X_train: npt.NDArray | None = None
    y_train: npt.NDArray | None = None
    groups: npt.NDArray | list | tuple | None = None
    one_hot_encoded: bool = False
    
    # Note: X_val is separate from the validation folds
    # ! TODO: Deprecate all X_val in favor of folds (where vanilla validation is just single fold of specified train-val split)
    # X_val: npt.NDArray | None = None
    # y_val: npt.NDArray | None = None
    
    X_test: npt.NDArray | None = None
    y_test: npt.NDArray | None = None
    
    # Should be a tuple of the form (train_inds, val_inds, test_inds), where each of the three elements is a list of indices or empty list/array.
    train_val_test_indicies: tuple | None = None
    
    # Ideally, folds is a list of dictionaries, where each dictionary contains the indices for the training and validations sets,
        # under keys 'train' and 'val', respectively.
    # NOTE: FOLDS SHOULD BE CREATED USING THE TRAINING DATA (i.e. X_train) ONLY
    folds: npt.NDArray | list | tuple | None = None
    metadata: dict | None = None
    
    def __init__(self, **data):
        super().__init__(**data)
        
        print('Assigning X_train and y_train to X and y, respectively... until train-test split is called.')
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
        print('Splitting data into training and testing sets...')
        self.X_train = self.X[train_inds]
        self.y_train = self.y[train_inds]
        self.X_test = self.X[test_inds]
        self.y_test = self.y[test_inds]

    # TODO: Define the following 'get_...' funcs as @property methods?
    def get_training_data(self):
        return self.X_train, self.y_train

    def get_testing_data(self):
        return self.X_test, self.y_test

    # def get_validation_data(self):
    #     return self.X_val, self.y_val
    
    def get_fold(self, fold_num):
        """
        Returns the training and testing data for the fold specified by fold_ind.
        """
        train_inds = self.folds[fold_num]['train']
        val_inds = self.folds[fold_num]['val']
        return self.X_train[train_inds], self.y_train[train_inds], self.X_train[val_inds], self.y_train[val_inds]
    
    def get_training_folds(self):
        """
        Returns the training data for all folds.
        """
        return [self.X_train[fold['train']] for fold in self.folds]
    
    def get_validation_folds(self):
        """
        Returns the testing data for all folds.
        """
        return [self.X_train[fold['val']] for fold in self.folds]
    
    def assign_train_val_test_indices(self, train_inds=[], val_inds=[], test_inds=[]):
        """
        Assigns the training, validation, and testing indicies to the dataclass fields.
        """
        self.train_val_test_indicies = (train_inds, val_inds, test_inds)
        
    
    # def train_val_test_split(self, train_inds, val_inds, test_inds):
    #     """
    #     Splits the data into training, validation, and testing sets based on the indices provided.

    #     Args:
    #         train_inds (list): A list of indices to use for the training set.
    #         val_inds (list): A list of indices to use for the validation set.
    #         test_inds (list): A list of indices to use for the testing set.

    #     Returns:
    #         None
    #     """
    #     self.X_train = self.X[train_inds]
    #     self.y_train = self.y[train_inds]
    #     self.X_val = self.X[val_inds]
    #     self.y_val = self.y[val_inds]
    #     self.X_test = self.X[test_inds]
    #     self.y_test = self.y[test_inds]


    # # ! Deprecated
    # def get_fold_deprecated(self, fold_num, fold_on='X'):
    #     """
    #     Returns the training and testing data for the fold specified by fold_ind.

    #     Args:
    #         fold_ind (int): The index of the fold to return.
    #         fold_on (str): The data to fold on. Either 'X' or 'X_train'. Defaults to 'X'.
    #             Folds are either performed on the entire dataset (X) or the training set (X_train).

    #     Returns:
    #         X_train (numpy.ndarray): The training data for the specified fold.
    #         y_train (numpy.ndarray): The training labels for the specified fold.
    #         X_test (numpy.ndarray): The testing data for the specified fold.
    #         y_test (numpy.ndarray): The testing labels for the specified fold.
    #     """
    #     if self.folds is None:
    #         raise ValueError('No folds have been defined. Please pass fold indicies to field "folds" to define folds.')
        
    #     fold_train_inds = self.folds[fold_num]
    #     if fold_on == 'X':
    #         fold_test_inds = np.setdiff1d(np.arange(len(self.X)), fold_train_inds)
    #         return self.X[fold_train_inds], self.y[fold_train_inds], self.X[fold_test_inds], self.y[fold_test_inds]
    #     elif fold_on == 'X_train':
    #         fold_test_inds = np.setdiff1d(np.arange(len(self.X_train)), fold_train_inds)
    #         return self.X[fold_train_inds], self.y[fold_train_inds], self.X[fold_test_inds], self.y[fold_test_inds]
    #     else:
    #         raise ValueError(f"Argument fold_on must be either 'X' or 'X_train'. {fold_on} is not recognized")
    