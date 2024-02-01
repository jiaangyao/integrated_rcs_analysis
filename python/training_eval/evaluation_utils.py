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
    confusion_matrix,
    classification_report,
)

import torch

# TODO: Probably should switch to torchmetrics.functional.<metric> ... these functions keep past states in memory...
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    AveragePrecision,
    CohenKappa,
    MatthewsCorrCoef,
    ConfusionMatrix,
    PrecisionRecallCurve,
)


def check_fitted(clf):
    return hasattr(clf, "classes_")


def get_num_classes(y_true, y_pred, y_pred_proba, one_hot_encoded):
    """
    Determines the number of classes based on the true and predicted labels.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
        one_hot_encoded (bool): Whether the labels are one-hot encoded.

    Returns:
        int: The number of classes.
    """
    # Take the max of the number of classes in y_true and y_pred,
    # in case all classes are not present in one
    if one_hot_encoded:
        num_classes = max(len(np.unique(y_true)), y_pred_proba.shape[-1])
    else:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
    return int(num_classes)


# Save confusion matrix as single number, for packing

# Define function to calculate desired scores
def custom_scorer_torch(y_true, y_pred, y_pred_proba, scoring, one_hot_encoded=False):
    """
    Computes various scoring metrics for the true and predicted labels.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
        scoring (list): A list of scoring metrics to compute.
        one_hot_encoded (bool): Whether the labels are one-hot encoded.

    Returns:
        dict: A dictionary of scoring metrics and their corresponding scores.
    """
    scoring = [score.lower() for score in scoring]
    scores = {}

    # Ensure y_true and y_pred are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true_tensor = torch.tensor(y_true)
    else:
        y_true_tensor = y_true

    # These metrics assume that y_true is not one-hot encoded
    if one_hot_encoded:  # Convert y_true to class labels
        y_true_tensor = torch.argmax(y_true_tensor, axis=-1)

    if not isinstance(y_pred, torch.Tensor):
        y_pred_tensor = torch.tensor(y_pred).to_device()
    else:
        y_pred_tensor = y_pred

    if not isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba_tensor = torch.tensor(y_pred_proba).to_device()
    else:
        y_pred_proba_tensor = y_pred_proba

    # Put all tensors on the same device
    y_true_tensor = y_true_tensor.detach().cpu()
    y_pred_tensor = y_pred_tensor.detach().cpu()
    y_pred_proba_tensor = y_pred_proba_tensor.detach().cpu()

    # Get number of classes
    num_classes = get_num_classes(y_true, y_pred, y_pred_proba, one_hot_encoded)
    if num_classes == 2:
        task = "binary"
    else:
        task = "multiclass"

    # Probability based metrics
    if ("roc_auc" in scoring or "auc" in scoring) and one_hot_encoded:
        roc_auc = AUROC(num_classes=num_classes, average="weighted", task=task)
        scores["roc_auc"] = roc_auc(y_pred_proba_tensor, y_true_tensor).item()

    if (
        "precision_recall_curve" in scoring
        or "prc" in scoring
        or "pr_curve" in scoring
        or "precisionrecallcurve" in scoring
        or "prcurve" in scoring
    ) and one_hot_encoded:
        raise NotImplementedError(
            "Precision recall curve is not yet implemented for one-hot encoded data."
        )
        # precision_recall_curve = PrecisionRecallCurve(num_classes=num_classes, task=task)
        # scores['precision_recall_curve'] = precision_recall_curve(y_pred_proba_tensor, y_true_tensor).numpy()

    # Prediction based metrics
    if "accuracy" in scoring or "acc" in scoring:
        accuracy = Accuracy(num_classes=num_classes, task=task)
        scores["accuracy"] = accuracy(y_pred_tensor, y_true_tensor).item()

    if "precision" in scoring:
        precision = Precision(average="weighted", num_classes=num_classes, task=task)
        scores["precision"] = precision(y_pred_tensor, y_true_tensor).item()

    if "recall" in scoring:
        recall = Recall(average="weighted", num_classes=num_classes, task=task)
        scores["recall"] = recall(y_pred_tensor, y_true_tensor).item()

    if "f1" in scoring:
        f1 = F1Score(average="weighted", num_classes=num_classes, task=task)
        scores["f1"] = f1(y_pred_tensor, y_true_tensor).item()

    if "average_precision" in scoring:
        average_precision = AveragePrecision(
            num_classes=num_classes, average="weighted", task=task
        )
        scores["average_precision"] = average_precision(
            y_pred_tensor, y_true_tensor
        ).item()

    if "cohen_kappa" in scoring or "kappa" in scoring or "cohenkappa" in scoring:
        cohen_kappa = CohenKappa(num_classes=num_classes, task=task)
        scores["cohen_kappa"] = cohen_kappa(y_pred_tensor, y_true_tensor).item()

    if (
        "matthews_corrcoef" in scoring
        or "mcc" in scoring
        or "matthewscorrcoef" in scoring
    ):
        matthews_corrcoef = MatthewsCorrCoef(num_classes=num_classes, task=task)
        scores["matthews_corrcoef"] = matthews_corrcoef(
            y_pred_tensor, y_true_tensor
        ).item()

    if "confusion_matrix" in scoring or "confusionmatrix" in scoring:
        confusion_matrix = ConfusionMatrix(num_classes=num_classes, task=task)
        scores["confusion_matrix"] = confusion_matrix(
            y_pred_tensor, y_true_tensor
        ).numpy()

    return scores


# Define function to calculate desired scores
def custom_scorer_sklearn(clf, X_val, y_true, scoring):
    assert check_fitted(clf)  # Ensure that the model is fitted

    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)
    scores = {}

    # Prediction based metrics
    if "accuracy" in scoring or "acc" in scoring or "ACC" in scoring:
        scores["accuracy"] = accuracy_score(y_true, y_pred)
    if "precision" in scoring:
        scores["precision"] = precision_score(y_true, y_pred, average="weighted")
    if "recall" in scoring:
        scores["recall"] = recall_score(y_true, y_pred, average="weighted")
    if "f1" in scoring:
        scores["f1"] = f1_score(y_true, y_pred, average="weighted")
    if "balanced_accuracy" in scoring:
        scores["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    if "cohen_kappa" in scoring:
        # ! Not debugged yet
        scores["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    if "matthews_corrcoef" in scoring:
        scores["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
    if "confusion_matrix" in scoring:
        scores["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Confidence/Probability based metrics
    if "roc_auc" in scoring or "auc" in scoring or "AUC" in scoring:
        scores["roc_auc"] = roc_auc_score(
            y_true,
            y_pred_proba,
        )
    if (
        "roc_auc_ovr" in scoring
        or "auc_ovr" in scoring
        or "AUC_OVR" in scoring
        or "AUC_ovr" in scoring
    ):
        scores["roc_auc_ovr"] = roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        )
    if (
        "roc_auc_ovo" in scoring
        or "auc_ovo" in scoring
        or "AUC_OVO" in scoring
        or "AUC_ovo" in scoring
    ):
        scores["roc_auc_ovo"] = roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovo"
        )
    if "average_precision" in scoring:
        scores["average_precision"] = average_precision_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        )

    if "classification_report" in scoring:
        # ! NOT DEBUGGED YET
        scores["classification_report"] = classification_report(y_true, y_pred)
    return scores
