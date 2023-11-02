from sklearn.model_selection import cross_validate, LeaveOneGroupOut, cross_val_score
from sklearn.model_selection._split import _BaseKFold
import sklearn.model_selection as skms
from sklearn.model_selection import train_test_split
from utils.pipeline_utils import convert_string_to_callable
from sklearn.model_selection import train_test_split
from training_eval.early_stopping import EarlyStopping
from utils.wandb_utils import wandb_log

import numpy as np

from sklearn.metrics import (
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
    # Partition data into relevant sets if necessary
    if early_stopping_conf := config.get("early_stopping"):
        early_stopping = EarlyStopping(**early_stopping_conf | {"val_obj": VanillaValidation()})
    else:
        early_stopping = None
        
    if config["data_split"]["name"] == "TrainTestSplit":
        train_inds, test_inds = train_test_split_inds(
            data_class.X,
            config['data_split']["test_size"], 
            config["random_seed"]
        )
        data_class.assign_train_val_test_indices(train_inds, [], test_inds)
        data_class.train_test_split(train_inds, test_inds)
    
    elif config["data_split"]["name"] == "TrainValidationTestSplit":
        train_inds, val_inds, test_inds = train_val_test_split_inds(
            data_class.X,
            config["data_split"]["validation_size"], 
            config["data_split"]["test_size"], 
            config["random_seed"]
        )
        data_class.assign_train_val_test_indices(train_inds, val_inds, test_inds)
        data_class.train_val_test_split(train_inds, val_inds, test_inds)

    # TODO: Expand to multiple validation methods? Could potentially be useful for comparing different methods... store as list?
    validation_name = list(config["validation_method"].keys())[0]
    if 'fold' in validation_name.lower() or 'group' in validation_name.lower():
        config["validation_method"]["random_state"] = config["random_seed"]
        val_object = set_up_cross_validation(config["validation_method"], VALIDATION_LIBRARIES)

        if data_class.one_hot_encoded:
            data_class.folds = [{'train': train_fold, 'val': test_fold}
                                for (train_fold, test_fold) 
                                in val_object.split(data_class.X_train, data_class.y_train.argmax(axis=1), groups=data_class.groups)]
        else:
            data_class.folds = [{'train': train_fold, 'val': test_fold}
                                for (train_fold, test_fold) 
                                in val_object.split(data_class.X_train, data_class.y_train, groups=data_class.groups)]
    else:
        val_object = None

    return ModelEvaluation(validation_name, val_object, config["scoring"], config["model_type"], config["random_seed"]), early_stopping

class VanillaValidation:
    """
    Class for vanilla validation (i.e. no cross-validation). Train on train set, validate on validation set.
    """
    def __init__(self) -> None:
        pass
    
    def get_scores_sklearn(self, model, X_train, y_train, X_val, y_val, scoring):
        model.fit(X_train, y_train)
        return custom_scorer(y_val, model.predict(X_val), scoring)
    
    
    # TODO: This has some redundancy with custom_scorer...
    def validation_scoring_torch(self, model_class, X_val, y_val, scoring):
        """
        Computes validation scores for a given model and validation data.

        Args:
            model_class (object): The model class to use for scoring.
            X_val (array-like): The validation input data.
            y_val (array-like): The validation target data.
            scoring (list): A list of scoring metrics to compute.

        Returns:
            dict: A dictionary of scoring metrics and their corresponding scores.
        """
        scores = {}
        

        if "accuracy" in scoring:
            scores["accuracy"] = model_class.get_accuracy(X_val, y_val)
            
        if "roc_auc" in scoring or "auc" in scoring or "AUC" in scoring:
            scores["roc_auc"] = model_class.get_auc(model_class.predict(X_val), y_val)
        
        return scores
            
            
    def get_scores_torch(self, model_class, X_train, y_train, X_val, y_val, scoring, one_hot_encoded, random_seed):
        """
        Trains a model and computes validation scores.

        Args:
            model_class (object): The model class to use for training and scoring.
            X_train (array-like): The training input data.
            y_train (array-like): The training target data.
            X_val (array-like): The validation input data.
            y_val (array-like): The validation target data.
            scoring (list): A list of scoring metrics to compute.
            early_stopping (bool): Whether to use early stopping during training.
            one_hot_encoded (bool): Whether the target data is one-hot encoded.
            random_seed (int): The random seed for splitting the data for early stopping.

        Returns:
            dict: A dictionary of scoring metrics and their corresponding scores.
        """
        vec_avg_loss, vec_avg_valid_loss = model_class.trainer.train(X_train, y_train)
        
        if vec_avg_valid_loss:
            wandb_log({"Average Epoch Loss": vec_avg_loss, "Average Epoch Validation Loss": vec_avg_valid_loss})
        else:
            wandb_log({"Average Epoch Loss": vec_avg_loss})
        
        if one_hot_encoded: y_val = np.argmax(y_val, axis=-1)
        
        model_class.model.eval()
        return self.validation_scoring(model_class, X_val, y_val, scoring)

# This model can be used as a base model, if people want a more specific evaluation class, they can create a new one
class ModelEvaluation:
    def __init__(self, validation_name, val_object, scoring, model_type, random_seed):
        self.validation_name = validation_name
        self.val_object = val_object
        self.scoring = scoring
        self.model_type = model_type
        self.random_seed = random_seed
        # TODO: IMPLEMENT EARLY STOPPING
        # self.eary_stopping = False
        self.determine_evaluation_method()

    # TODO: Check if model is instance of sklearn or torch model, then call appropriate evaluation function with relevant scoring
    # TODO: If torch model but evaluation is sklearn, then wrap torch model in skorch
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

    # Should evaluate_model take in model and data_class??
    # Evaluate should include training with prediction, for just prediction then use get_scores
    def evaluate_model(self, model_class, data_class):
        # Note: This function first fits then evaluates the model.
        # To just get scores on an already fit model, use get_scores()
        return self.evaluate_model_specific(model_class, data_class)


    # TODO: Figure out if I need groups param if I set up LeaveOneGroupOut.split(X,y groups) in uncostrained_pipeline.py -> set_up_cross_validation
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
        # Evaluate predictions
        X_train, y_train = data_class.get_training_data()
        groups = data_class.groups
        model = model_class.model

        if isinstance(self.val_object, _BaseKFold):
            results = cross_validate(
                model,
                X_train,
                y_train,
                cv=self.val_object,
                scoring=self.scoring,
                n_jobs=self.val_object.get_n_splits() + 1,
            )
        elif isinstance(self.val_object, LeaveOneGroupOut):
            # May need to pass groups as param
            results = cross_val_score(
                model,
                X_train,
                y_train,
                cv=self.val_object,
                groups=groups,
                scoring=self.scoring,
                n_jobs=self.val_object.get_n_splits() + 1,
            )
        elif isinstance(self.val_object, VanillaValidation):
            # For vanilla prediction on y_train
            X_val, y_val = data_class.get_validation_data()
            results = self.val_object.get_scores_sklearn(model, X_train, y_train, X_val, y_val, self.scoring)
        else:
            raise ValueError(f"Validation object {self.val_object} is not recognized.")
        return results

    # TODO: Figure out parallization for torch
    # TODO: Figure out location to handle one hot encoding vs multiclass for accuracy scores
    def evaluate_model_torch(self, model_class, data_class):
        scores = {}
        vanilla_validation = VanillaValidation()
        
        for key in self.scoring: scores[key] = []
        
        for i in range(len(data_class.folds)):
            # NOTE: Each fold is treated as a vanilla validation 
            X_train, y_train, X_val, y_val = data_class.get_fold(i)

            model_class.reset_model()
            
            [scores[key].append(score) for key, score 
            in vanilla_validation.get_scores_torch(model_class, X_train, y_train, X_val, y_val, self.scoring, data_class.one_hot_encoded, self.random_seed).items()]
            
        # scores = {f"{key}_mean": np.mean(scores[key]) for key in scores} | {f"{key}_std": np.std(scores[key]) for key in scores}
        return scores
    
        
    def get_scores(self, model, X, y, one_hot_encoded=False):
        if self.is_torch:
            if hasattr(model.module, "predict"):
                if one_hot_encoded:
                    return custom_scorer(y, np.argmax(model.predict(X), axis=-1), self.scoring)
                else:
                    return custom_scorer(y, model.predict(X), self.scoring)
            else:
                print("No predict method found, using forward method instead.")
                return custom_scorer(y, model.forward(X), self.scoring)
        else:
            return custom_scorer(y, model.predict(X), self.scoring)
