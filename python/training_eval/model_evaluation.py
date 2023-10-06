from sklearn.model_selection import cross_validate, LeaveOneGroupOut, cross_val_score
from sklearn.model_selection._split import _BaseKFold
import sklearn.model_selection as skms
from utils.pipeline_utils import convert_string_to_callable

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
    if "accuracy" in scoring:
        scores["accuracy"] = accuracy_score(y_true, y_pred)
    if "precision" in scoring:
        scores["precision"] = precision_score(y_true, y_pred, average="weighted")
    if "recall" in scoring:
        scores["recall"] = recall_score(y_true, y_pred, average="weighted")
    if "f1" in scoring:
        scores["f1"] = f1_score(y_true, y_pred, average="weighted")
    if "roc_auc" in scoring:
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


def set_up_cross_validation(config_dict, libs):
    cross_val_type = list(config_dict.keys())[0]
    func = convert_string_to_callable(libs, cross_val_type)
    if type(func) is str:
        return func
    else:
        kwargs = list(config_dict.values())[0]
        return func(**kwargs)


def create_eval_class_from_config(config):
    # TODO: Update with new evaluation config format
    validation = set_up_cross_validation(config["method"], VALIDATION_LIBRARIES)
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
        # TODO: Implement fitting then evaluation of model on X_train, y_train
        pass

    def get_scores(self, model, X, y):
        if self.is_torch:
            if hasattr(model.module, "predict"):
                return custom_scorer(y, model.predict(X), self.scoring)
            else:
                print("No predict method found, using forward method instead.")
                return custom_scorer(y, model.forward(X), self.scoring)
        else:
            return custom_scorer(y, model.predict(X), self.scoring)
