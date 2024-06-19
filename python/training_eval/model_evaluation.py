from sklearn.model_selection import (
    cross_validate,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.model_selection._split import _BaseKFold
import sklearn.model_selection as skms
from sklearn.model_selection import train_test_split, LeavePGroupsOut
from utils.pipeline_utils import convert_string_to_callable
from sklearn.model_selection import train_test_split
from utils.wandb_utils import wandb_log
import wandb

import numpy as np

# Libraries for (cross) validation
import sklearn.model_selection as skms

from .evaluation_utils import custom_scorer_sklearn, custom_scorer_torch

VALIDATION_LIBRARIES = [skms]


def train_test_split_inds(
    X, test_size=0.2, random_seed=42, shuffle=True, stratify=None
):
    """
    Split data into train, and test sets.
    - data (array-like): The data to be split.
    - test_ratio (float): The ratio of the test set. Default is 0.2.
    - random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
    - inds_X_train (array-like): The indices of the training set.
    - inds_X_test (array-like): The indices of the test set.
    """
    # Manage parameter edge cases
    if shuffle is None:
        shuffle = False
    if stratify is not None:
        shuffle = True

    inds = np.arange(len(X))
    try:
        inds_train, inds_test = train_test_split(
            inds,
            test_size=test_size,
            random_state=random_seed,
            shuffle=shuffle,
            stratify=stratify,
        )
    except ValueError as e:
        print(e)
        print("Train test split tries to split on 0th dimension. Try having your 0th dimension be the number of samples...")
    return inds_train, inds_test


def train_val_test_split_inds(
    X, val_size=0.2, test_size=0.2, random_seed=42, shuffle=True, stratify=False
):
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
    assert (
        val_size + test_size < 1
    ), "Validation and test ratios combined must be less than 1"
    if shuffle is None:
        shuffle = False

    train_size = 1 - val_size - test_size

    # Split data into train and temp (temp will be split into val and test)
    inds_X_train, inds_tmp = train_test_split(
        np.arange(len(X)),
        test_size=1 - train_size,
        random_state=random_seed,
        shuffle=shuffle,
        stratify=stratify,
    )

    # Calculate validation split ratio from remaining data
    val_split = val_size / (val_size + test_size)

    # Split temp_data into validation and test data
    inds_X_val, inds_X_test = train_test_split(
        inds_tmp, test_size=1 - val_split, random_state=random_seed, shuffle=shuffle
    )

    return inds_X_train, inds_X_val, inds_X_test


def set_up_cross_validation(config_dict, libs):
    cross_val_type = list(config_dict.keys())[0]
    func = convert_string_to_callable(libs, cross_val_type)
    if type(func) is str:
        return func
    else:
        kwargs = list(config_dict.values())[0]
        if kwargs is None:
            kwargs = {}
        return func(**kwargs)


def implement_train_test_split(config, data_class, logger=None):
    """
    Implements the train-test split based on the configuration parameters.

    Args:
        config (dict): Configuration parameters for the train-test split.
        data_class (object): Data class object containing the data and labels.

    Returns:
        None
    """

    if config["data_split"]["name"] == "TrainTestSplit":
        if config["data_split"].get("stratify") is True:
            stratify = data_class.y
        train_inds, test_inds = train_test_split_inds(
            data_class.X,
            config["data_split"]["test_size"],
            config["random_seed"],
            shuffle=config["data_split"].get("shuffle"),
            stratify=stratify,
        )
        data_class.assign_train_val_test_indices(train_inds, [], test_inds)
        logger.info("Splitting data into training and testing sets...")
        data_class.train_test_split(train_inds, test_inds)

    elif config["data_split"]["name"] == "LeavePGroupsOut":
        lpgo = LeavePGroupsOut(n_groups=config["data_split"]["n_groups"])
        # Assumes a random subset of groups are being held-out, so only take first fold
        # Further grouping splits should be handled by LeaveOneGroupOut cross-validation object
        if not config["data_split"]["select_random"]:
            train_inds, test_inds = next(
                lpgo.split(data_class.X, groups=data_class.groups)
            )
        else:  # Select random fold
            # Generate a random index
            random_index = np.random.RandomState(config["random_seed"]).randint(
                0, lpgo.get_n_splits(groups=data_class.groups) - 1
            )

            # Iterate over the generator until you reach the random index
            for i, (train_tmp, test_tmp) in enumerate(
                lpgo.split(data_class.X, groups=data_class.groups)
            ):
                if i == random_index:
                    train_inds, test_inds = train_tmp, test_tmp

        data_class.assign_train_val_test_indices(train_inds, [], test_inds)
        logger.info("Splitting data into training and testing sets...")
        data_class.train_test_split(train_inds, test_inds)

    if logger is not None:
        logger.info(f"Train Size: {len(train_inds)}")
        logger.info(f"Test Size: {len(test_inds)}")

    return None


def set_up_training_validation_folds(config, data_class):
    """
    Sets up the training and validation folds for model evaluation. Creates fold indices for training data and stores them in the data class object.
    Ignores test (i.e. hold-out) data.

    Args:
        config (dict): Configuration parameters for setting up the folds.
        data_class (object): Object containing the training and validation data.

    Returns:
        tuple: A tuple containing the name of the validation method and the validation object.

    Raises:
        None

    """

    # TODO: Expand to multiple (i.e. more than one) validation methods? Could potentially be useful for comparing different methods... store as list?

    # In the case that no validation method is specified, return None. This may be useful in the case where a data_class with folds is imported into the main pipeline.
    if config.get("validation_method") is None:
        return None, None

    validation_name = list(config["validation_method"].keys())[0]
    if (
        "fold" in validation_name.lower() or "group" in validation_name.lower()
    ):  # StratifiedKFold, GroupKFold, etc...
        config["validation_method"]["random_state"] = config["random_seed"]
        val_object = set_up_cross_validation(
            config["validation_method"], VALIDATION_LIBRARIES
        )

        if data_class.one_hot_encoded:
            y_tmp = data_class.y_train.argmax(axis=1)
        else:
            y_tmp = data_class.y_train

        if data_class.groups is None:
            data_class.folds = [
                {"train": train_fold, "val": test_fold}
                for (train_fold, test_fold) in val_object.split(
                    data_class.X_train, y_tmp
                )
            ]
        else:
            data_class.folds = [
                {
                    "train": train_fold,
                    "val": test_fold,
                }
                for (train_fold, test_fold) in val_object.split(
                    data_class.X_train, y_tmp, groups=data_class.groups_train
                )
            ]

    # Treating VanillaValidation as special case of 1 fold cross validation
    elif "vanilla" in validation_name.lower():
        vanilla_val_config = config["validation_method"].get(validation_name)
        train_fold, test_fold = train_test_split_inds(
            data_class.X_train,
            vanilla_val_config["validation_size"],
            random_seed=config["random_seed"],
        )
        data_class.folds = [{"train": train_fold, "val": test_fold}]
        val_object = None

    return validation_name, val_object


def create_eval_class_from_config(config, data_class):

    validation_name, val_object = set_up_training_validation_folds(config, data_class)

    return ModelEvaluation(
        validation_name,
        val_object,
        config["scoring"],
        config["model_type"],
        config["random_seed"],
    )


class VanillaValidation:
    """
    Class for vanilla validation (i.e. no cross-validation). Train on train set, validate on validation set.
    """

    def __init__(self) -> None:
        pass

    def get_scores_sklearn(self, model_class, X_train, y_train, X_val, y_val, scoring):
        model_class.model.fit(X_train, y_train)
        return custom_scorer_sklearn(model_class.model, X_val, y_val, scoring)

    def get_scores_torch(
        self, model_class, X_train, y_train, X_val, y_val, scoring, one_hot_encoded
    ):
        """
        Trains a model and computes validation scores.

        Args:
            model_class (object): The model class to use for training and scoring.
            X_train (array-like): The training input data.
            y_train (array-like): The training target data.
            X_val (array-like): The validation input data.
            y_val (array-like): The validation target data.
            scoring (list): A list of scoring metrics to compute.
            one_hot_encoded (bool): Whether the target data is one-hot encoded.
        Returns:
            dict: A dictionary of scoring metrics and their corresponding scores.
        """
        print("Training model...")
        (
            epoch_avg_loss,
            epoch_scores,
            epoch_avg_valid_loss,
            epoch_val_scores,
        ) = model_class.trainer.train(X_train, y_train, one_hot_encoded)
        metrics_by_epoch = {"Epoch Train Loss": epoch_avg_loss} | {
            "Epoch Train " + key: epoch_scores[key] for key in epoch_scores.keys()
        }

        if epoch_avg_valid_loss is not None:
            metrics_by_epoch |= {"Epoch Val Loss": epoch_avg_valid_loss}
        if epoch_val_scores is not None:
            metrics_by_epoch |= {
                "Epoch Val " + key: epoch_val_scores[key]
                for key in epoch_val_scores.keys()
            }

        model_class.model.eval()
        y_pred_proba = model_class.predict_proba(X_val)
        y_pred = model_class.predict_class(X_val)

        return (
            custom_scorer_torch(y_val, y_pred, y_pred_proba, scoring, one_hot_encoded),
            metrics_by_epoch,
        )


# This model can be used as a base model, if people want a more specific evaluation class, they can create a new one
class ModelEvaluation:
    def __init__(self, validation_name, val_object, scoring, model_type, random_seed):
        self.validation_name = validation_name
        self.val_object = val_object
        self.scoring = scoring
        self.model_type = model_type
        self.random_seed = random_seed  # Likely to be deprecated
        self.determine_evaluation_method()
        self.determine_test_method()

    def determine_evaluation_method(self):
        if self.model_type == "torch":
            self.evaluate_model_specific = self.evaluate_model_torch
            self.is_torch = True
        elif self.model_type == "sklearn" or self.model_type == "skorch":
            self.evaluate_model_specific = self.evaluate_model_sklearn
            self.is_torch = False
        else:
            raise ValueError(
                f"Model {self.model_type} is not recognized for determining evaulation method."
            )

    def evaluate_model(self, model_class, data_class):
        # Note: This function first fits then evaluates the model.
        # To just get scores on an already fit model, use get_scores()
        return self.evaluate_model_specific(model_class, data_class)

    def determine_test_method(self):
        if self.model_type == "torch":
            self.test_model_specific = self.test_model_torch
            self.is_torch = True
        elif self.model_type == "sklearn" or self.model_type == "skorch":
            self.test_model_specific = self.test_model_sklearn
            self.is_torch = False
        else:
            raise ValueError(
                f"Model {self.model_type} is not recognized for determining evaulation method."
            )

    def test_model(self, model_class, data_class):
        # Note: This function first fits then tests the model on test set.
        # To just get scores on an already fit model, use get_scores()
        return self.test_model_specific(model_class, data_class)

    def evaluate_model_sklearn(self, model_class, data_class):
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
        # TODO: Parameterize n_jobs

        scores = {}

        for key in self.scoring:
            scores[key] = []

        # TODO: Figure out how to parallelize this for loop
        for i in range(len(data_class.folds)):

            X_train, y_train, X_val, y_val = data_class.get_fold(i)

            model_class.reset_model()

            model_class.model.fit(X_train, y_train)

            fold_scores = custom_scorer_sklearn(
                model_class.model, X_val, y_val, self.scoring
            )

            # Collect scores
            [scores[key].append(score) for key, score in fold_scores.items()]

        return scores, {}  # Empty lists for epoch losses and validation losses

    # TODO: Figure out parallization for torch. Potentially consolidate with sklearn, then parallelization only need to be solved once?
    # TODO: Alternatively, use Ray for Torch and Joblib for sklearn
    def evaluate_model_torch(self, model_class, data_class):
        scores = {}
        metrics_by_epoch = {}
        vanilla_validation = VanillaValidation()

        for key in self.scoring:
            scores[key] = []

        for i in range(len(data_class.folds)):
            # NOTE: Each fold is treated as a vanilla validation
            X_train, y_train, X_val, y_val = data_class.get_fold(i)

            # Reset model, so that it starts from scratch for each fold
            model_class.reset_model()

            fold_scores, fold_metrics_by_epoch = vanilla_validation.get_scores_torch(
                model_class,
                X_train,
                y_train,
                X_val,
                y_val,
                self.scoring,
                data_class.one_hot_encoded,
            )

            # Collect scores
            [scores[key].append(score) for key, score in fold_scores.items()]

            [
                metrics_by_epoch.setdefault(key, [])
                for key in fold_metrics_by_epoch.keys()
            ]
            [
                metrics_by_epoch[key].append(score)
                for key, score in fold_metrics_by_epoch.items()
            ]

        # scores = {f"{key}_mean": np.mean(scores[key]) for key in scores} | {f"{key}_std": np.std(scores[key]) for key in scores}
        return scores, metrics_by_epoch

    def test_model_sklearn(self, model_class, data_class):
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

        scores = {}

        for key in self.scoring:
            scores[key] = []

        X_train, y_train = data_class.get_training_data()
        X_test, y_test = data_class.get_testing_data()

        model_class.model.fit(X_train, y_train)

        fold_scores = custom_scorer_sklearn(
            model_class.model, X_test, y_test, self.scoring
        )

        # Collect scores
        [scores[key].append(score) for key, score in fold_scores.items()]

        return scores, {}  # Empty lists for epoch losses and validation losses

    def test_model_torch(self, model_class, data_class):
        scores = {}
        metrics_by_epoch = {}
        vanilla_validation = VanillaValidation()

        for key in self.scoring:
            scores[key] = []

        X_train, y_train = data_class.get_training_data()
        X_test, y_test = data_class.get_testing_data()

        # Train and get test
        fold_scores, fold_metrics_by_epoch = vanilla_validation.get_scores_torch(
            model_class,
            X_train,
            y_train,
            X_test,
            y_test,
            self.scoring,
            data_class.one_hot_encoded,
        )

        # Collect scores
        [scores[key].append(score) for key, score in fold_scores.items()]

        [metrics_by_epoch.setdefault(key, []) for key in fold_metrics_by_epoch.keys()]
        [
            metrics_by_epoch[key].append(score)
            for key, score in fold_metrics_by_epoch.items()
        ]

        # scores = {f"{key}_mean": np.mean(scores[key]) for key in scores} | {f"{key}_std": np.std(scores[key]) for key in scores}
        return scores, metrics_by_epoch

    # NOT DEBUGGED
    def get_scores(self, model, X, y, one_hot_encoded=False):
        # Example use case: Get scores for a model that has already been trained, e.g. on a test set
        if self.is_torch:
            if hasattr(model.module, "predict"):
                if one_hot_encoded:
                    return custom_scorer_torch(
                        y, np.argmax(model.predict(X), axis=-1), self.scoring
                    )
                else:
                    return custom_scorer_torch(y, model.predict(X), self.scoring)
            else:
                print("No predict method found, using forward method instead.")
                return custom_scorer_torch(y, model.forward(X), self.scoring)
        else:
            return custom_scorer_sklearn(y, model.predict(X), self.scoring)
