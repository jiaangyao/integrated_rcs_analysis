from sklearn.model_selection import cross_validate, LeaveOneGroupOut, cross_val_score
from sklearn.model_selection._split import _BaseKFold
import sklearn.model_selection as skms
from sklearn.model_selection import train_test_split
from utils.pipeline_utils import convert_string_to_callable

import numpy as np

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    classification_report,
)

# Libraries for (cross) validation
import sklearn.model_selection as skms

VALIDATION_LIBRARIES = [skms]

# Define function to calculate desired scores
def custom_scorer(y_true, y_pred, scoring):
    scores = {}
    if "accuracy" in scoring or "acc" in scoring or "ACC" in scoring:
        scores["accuracy"] = accuracy_score(y_true, y_pred)
    if "precision" in scoring:
        scores["precision"] = precision_score(y_true, y_pred, average="weighted")
    if "recall" in scoring:
        scores["recall"] = recall_score(y_true, y_pred, average="weighted")
    if "f1" in scoring:
        scores["f1"] = f1_score(y_true, y_pred, average="weighted")
    if "roc_auc" in scoring or "auc" in scoring or "AUC" in scoring:
        scores["roc_auc"] = roc_auc_score(
            y_true, y_pred, average="weighted", multi_class="ovr"
        )
    if "average_precision" in scoring:
        scores["average_precision"] = average_precision_score(
            y_true, y_pred, average="weighted", multi_class="ovr"
        )
    if "balanced_accuracy" in scoring:
        scores["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    if "cohen_kappa" in scoring:
        scores["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    if "matthews_corrcoef" in scoring:
        scores["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
    if "confusion_matrix" in scoring:
        scores["confusion_matrix"] = multilabel_confusion_matrix(y_true, y_pred)
    if "classification_report" in scoring:
        scores["classification_report"] = classification_report(y_true, y_pred)
    return scores


def train_test_split_inds(X, test_size=0.2, random_seed=42, shuffle=True):
        """
        Split data into train, and test sets.
        - data (array-like): The data to be split.
        - test_ratio (float): The ratio of the test set. Default is 0.2.
        - random_seed (int, optional): Random seed for reproducibility. Default is 42.
        
        Returns:
        - inds_X_train (array-like): The indices of the training set.
        - inds_X_test (array-like): The indices of the test set.
        """
        inds = np.arange(len(X))
        inds_train, inds_test = train_test_split(
            inds, test_size=test_size, random_state=random_seed, shuffle=shuffle
        )
        return inds_train, inds_test
        
    
def train_val_test_split_inds(X, val_size=0.2, test_size=0.2, random_seed=42, shuffle=True):
    """
    Split data into train, validation, and test sets using scikit-learn's train_test_split.

    Parameters:
    - data (array-like): The data to be split.
    - val_ratio (float): The ratio of the validation set. Default is 0.2.
    - test_ratio (float): The ratio of the test set. Default is 0.2.
    - random_seed (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
    - inds_X_train (array-like): The indices of the training set.
    - inds_X_val (array-like): The indices of the validation set.
    - inds_X_test (array-like): The indices of the test set.
    """
    
    # Make sure val_ratio and test_ratio are valid
    assert 0 <= val_size < 1, "Validation ratio must be between 0 and 1"
    assert 0 <= test_size < 1, "Test ratio must be between 0 and 1"
    assert val_size + test_size < 1, "Validation and test ratios combined must be less than 1"

    train_size = 1 - val_size - test_size

    # Split data into train and temp (temp will be split into val and test)
    inds_X_train, inds_tmp = train_test_split(
        np.arange(len(X)), test_size=1-train_size, random_state=random_seed, shuffle=shuffle
    )

    # Calculate validation split ratio from remaining data
    val_split = val_size / (val_size + test_size)
    
    # Split temp_data into validation and test data
    inds_X_val, inds_X_test = train_test_split(
        inds_tmp, test_size=1-val_split, random_state=random_seed, shuffle=shuffle
    )
    
    return inds_X_train, inds_X_val, inds_X_test


def set_up_cross_validation(config_dict, libs):
    cross_val_type = list(config_dict.keys())[0]
    func = convert_string_to_callable(libs, cross_val_type)
    if type(func) is str:
        return func
    else:
        kwargs = list(config_dict.values())[0]
        return func(**kwargs)


def create_eval_class_from_config(config, data_class):
    # TODO: Update with new evaluation config format
    # Partition data into relevant sets if necessary
    if config["method"] == "TrainTestSplit":
        train_inds, test_inds = train_test_split_inds(
            data_class.X,
            config["TrainTestSplit"]["test_size"], 
            config["random_seed"]
        )
        data_class.train_test_split(train_inds, test_inds)
        validation = None
    elif config["method"] == "TrainValidationTestSplit":
        train_inds, val_inds, test_inds = train_val_test_split_inds(
            data_class.X,
            config["TrainValidationTestSplit"]["validation_size"], 
            config["TrainValidationTestSplit"]["test_size"], 
            config["random_seed"]
        )
        data_class.train_val_test_split(train_inds, val_inds, test_inds)
        validation = None
        raise NotImplementedError # TODO: Figure out consistent validation object for both train-test split and cross validation
    elif 'Fold' in config["method"]:
        validation = set_up_cross_validation(config["method"], VALIDATION_LIBRARIES)
        # TODO: Include indices in data_class.folds that match what the validation object is doing in sklearn cross_validate.
        # TODO: Somehow handle 'fold_on' param... 
    return ModelEvaluation(validation, config["scoring"], config["name"])


class ModelEvaluation:
    def __init__(self, validation, scoring, module_type):
        self.validation = validation
        self.scoring = scoring
        self.module_type = module_type
        self.determine_evaluation_method()

    # TODO: Check is model is instance of sklearn or torch model, then call appropriate evaluation function with relevant scoring
    # TODO: If torch model but evaluation is sklearn, then wrap torch model in skorch
    def determine_evaluation_method(self):
        if self.module_type == "torch":
            self.evaluate_model_specific = self.evaluate_model_torch
            self.is_torch = True
        elif self.module_type == "sklearn" or self.module_type == "skorch":
            self.evaluate_model_specific = self.evaluate_model_sklearn
            self.is_torch = False
        else:
            raise ValueError(
                f"Model {self.module_type} is not recognized for determining evaulation method."
            )

    def evaluate_model(self, model, X_train, y_train):
        # Note: This function first fits then evaluates the model.
        # To just get scores on an already fit model, use get_scores()
        return self.evaluate_model_specific(model, X_train, y_train)

    # TODO: Figure out if I need groups param if I set up LeaveOneGroupOut.split(X,y groups) in uncostrained_pipeline.py -> set_up_cross_validation
    def evaluate_model_sklearn(self, model, X_train, y_train, groups=None):
        """
        Evaluate a machine learning model using cross-validation and return the results.

        Parameters:
        model (object): The machine learning model to evaluate.
        X_train: The training data features.
        y_train: The training data labels.
        validation_obj: The cross-validation object to use for evaluation.
        scoring: The scoring metric(s) to use for evaluation.
        groups: The group labels to use for group-based cross-validation (optional).

        Returns:
        dict: A
        """
        if model is None:
            model = self.model

        # TODO: Parameterize n_jobs
        # Evaluate predictions

        if isinstance(self.validation, _BaseKFold):
            results = cross_validate(
                model,
                X_train,
                y_train,
                cv=self.validation,
                scoring=self.scoring,
                n_jobs=self.validation.get_n_splits() + 1,
            )
        elif isinstance(self.validation, LeaveOneGroupOut):
            # May need to pass groups as param
            results = cross_val_score(
                model,
                X_train,
                y_train,
                cv=self.validation,
                groups=groups,
                scoring=self.scoring,
                n_jobs=self.validation.get_n_splits() + 1,
            )
        elif self.validation is None:
            # For vanilla prediction on y_train
            results = custom_scorer(y_train, model.predict(X_train), self.scoring)
        else:
            raise ValueError(f"Validation object {self.validation} is not recognized.")
        return results

    def evaluate_model_torch(self, model, X_train, y_train):
        scores = {}
        model.trainer.fit(X_train, y_train)
        
        if "accuracy" in self.scoring:
            scores["accuracy"] = model.model.get_accuracy(X_train, y_train)
            
        if "roc_auc" in self.scoring or "auc" in self.scoring or "AUC" in self.scoring:
            scores["roc_auc"] = model.model.get_auc(X_train, y_train)
        
        # TODO: Handle other metrics
        return scores
        
    def get_scores(self, model, X, y):
        if self.is_torch:
            if hasattr(model.module, "predict"):
                return custom_scorer(y, model.predict(X), self.scoring)
            else:
                print("No predict method found, using forward method instead.")
                return custom_scorer(y, model.forward(X), self.scoring)
        else:
            return custom_scorer(y, model.predict(X), self.scoring)
