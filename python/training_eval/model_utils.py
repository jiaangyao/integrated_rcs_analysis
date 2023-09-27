from sklearn.model_selection import cross_validate, LeaveOneGroupOut, cross_val_score
from sklearn.model_selection._split import _BaseKFold

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, \
    matthews_corrcoef, multilabel_confusion_matrix, classification_report

# Define function to calculate desired scores
def custom_scorer(y_true, y_pred, scoring):
    scores = {}
    if 'accuracy' in scoring:
        scores['accuracy'] = accuracy_score(y_true, y_pred)
    if 'precision' in scoring:
        scores['precision'] = precision_score(y_true, y_pred, average='weighted')
    if 'recall' in scoring:
        scores['recall'] = recall_score(y_true, y_pred, average='weighted')
    if 'f1' in scoring:
        scores['f1'] = f1_score(y_true, y_pred, average='weighted')
    if 'roc_auc' in scoring:
        scores['roc_auc'] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    if 'average_precision' in scoring:
        scores['average_precision'] = average_precision_score(y_true, y_pred, average='weighted', multi_class='ovr')
    if 'balanced_accuracy' in scoring:
        scores['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    if 'cohen_kappa' in scoring:
        scores['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    if 'matthews_corrcoef' in scoring:
        scores['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    if 'confusion_matrix' in scoring:
        scores['confusion_matrix'] = multilabel_confusion_matrix(y_true, y_pred)
    if 'classification_report' in scoring:
        scores['classification_report'] = classification_report(y_true, y_pred)
    return scores

# TODO: Set up n_jobs for parallelization
# TODO: Figure out if I need groups param if I set up LeaveOneGroupOut.split(X,y groups) in
# uncostrained_pipeline.py -> set_up_cross_validation 
def evaluate_model(model: object, X_train, y_train, validation_obj, scoring, groups=None):
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
    # Evaluate predictions
    if isinstance(validation_obj, _BaseKFold):
        results = cross_validate(model, X_train, y_train, cv=validation_obj, scoring=scoring)
    elif isinstance(validation_obj, LeaveOneGroupOut) and groups is not None:
        # May need to pass groups as param
        results = cross_val_score(model, X_train, y_train, cv=validation_obj, scoring=scoring)
    elif validation_obj is None:
        # For vanilla prediction on y_train
        results = custom_scorer(y_train, model.predict(X_train), scoring)
    else:
        raise ValueError(f"Validation object {validation_obj} is not recognized.")
    return results