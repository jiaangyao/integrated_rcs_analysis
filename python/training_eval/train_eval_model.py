import copy

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import normalize

from preproc.label_base import combine_labels


def step_model(
    model,
    vec_features: list[npt.NDArray],
    vec_y_class: list[npt.NDArray],
    y_stim_test: npt.NDArray,
    trainer_cfg: DictConfig | dict,
    n_class: int = 4,
    hashmap: dict[str, int] | None = None,
    # bool_return_pred: bool = False,
) -> dict[str, npt.NDArray | int | float | None]:
    """Train and evaluate a model on the provided features a list of [train, (valid), test] datasets

    Args:
        model (_type_): input model
        vec_features (list[npt.NDArray]): list of [train, (valid), test] datasets. Valid dataset is optional.
        vec_y_class (list[npt.NDArray]): list of [train, (valid), test] labels. Valid dataset is optional.
        y_stim_test (npt.NDArray): test set label for stim type
        trainer_cfg (DictConfig | dict): configuration for the trainer
        n_class (int, optional): number of class labels. Defaults to 4.
        hashmap (dict[str, int] | None, optional): dict mapping for primary label and secondary label. Defaults to None.

    Returns:
        dict[str, npt.NDArray | int | float | None]: output results
    """

    # unpack the variable first
    if model.bool_torch:
        feature_train, feature_valid, feature_test = vec_features
        y_class_train, y_class_valid, y_class_test = vec_y_class

        # if asked to normalize the features
        # TODO: integrate this into the DataLoader class in torch
        if trainer_cfg['bool_whiten_feature']:
            # calculate mean and std from training set
            mean_train = np.mean(feature_train, axis=0)
            sigma_train = np.std(feature_train, axis=0)

            # apply normalization to all sets
            feature_train = (feature_train - mean_train) / sigma_train
            feature_valid = (feature_valid - mean_train) / sigma_train
            feature_test = (feature_test - mean_train) / sigma_train

        # optionally perform L2 normalization
        # TODO: verify this function
        if trainer_cfg['bool_l2_normalize_feature']:
            # calculate l2 norm from training set
            l2_norm_train = np.linalg.norm(feature_train, axis=1)

            # apply normalization to all sets
            feature_train = feature_train / l2_norm_train[:, None]
            feature_valid = feature_valid / l2_norm_train[:, None]
            feature_test = feature_test / l2_norm_train[:, None]

        # TODO: get rid of following line, trainer_cfg should not be a dict
        if OmegaConf.is_config(trainer_cfg):
            trainer_cfg_model = OmegaConf.to_container(trainer_cfg, resolve=True)
        else:
            trainer_cfg_model = copy.deepcopy(trainer_cfg)
        trainer_cfg_model.pop('bool_whiten_feature')
        trainer_cfg_model.pop('bool_l2_normalize_feature')

    else:
        feature_train, feature_test = vec_features
        y_class_train, y_class_test = vec_y_class

        feature_valid = None
        y_class_valid = None
        trainer_cfg_model = None

    # now perform training
    model = train_model(model, feature_train, y_class_train, feature_valid, y_class_valid, trainer_cfg_model)

    # now perform evaluation
    output = eval_model(model, feature_test, y_class_test, y_stim_test, n_class, hashmap)

    return output


def train_model(model, feature_train, y_class_train, feature_valid=None, y_class_valid=None, trainer_cfg_model=None,):

    # now train the model
    # if input is torch model
    if model.bool_torch:
        model.train(
            feature_train,
            y_class_train,
            valid_data=feature_valid,
            valid_label=y_class_valid,
            **trainer_cfg_model
        )

    # if using sklearn models
    else:
        # now obtain the models
        model.train(feature_train, y_class_train)

    return model


def eval_model(model, features_test, y_class_test, y_stim_test, n_class, hashmap):
    # next generate the predictions
    y_class_pred = model.predict(features_test)
    if np.ndim(y_class_pred) > 1:
        y_class_pred = np.argmax(y_class_pred, axis=1)

    # estimate the ROC curve
    y_class_pred_scores = model.predict_proba(features_test)

    # compute metrics outside of if
    acc = accuracy_score(y_class_test, y_class_pred)
    f1 = f1_score(y_class_test, y_class_pred)

    # create the big confusion matrix
    full_label_test, _ = combine_labels(y_class_test, y_stim_test, hashmap=hashmap)
    full_label_pred, _ = combine_labels(y_class_pred, y_stim_test, hashmap=hashmap)
    if hashmap is not None:
        conf_mat = confusion_matrix(
            full_label_test, full_label_pred, labels=list(hashmap.values())
        )
    else:
        conf_mat = confusion_matrix(full_label_test, full_label_pred)

    # sanity check
    assert np.all(
        np.array(conf_mat.shape) == n_class
    ), "Confusion matrix is not the right size"

    # obtain the ROC curve
    # roc = roc_curve(y_class_test, y_class_pred_scores)
    if model.bool_torch:
        auc = model.get_auc(y_class_pred_scores, y_class_test)
    else:
        auc = roc_auc_score(y_class_test, y_class_pred_scores)

    # create the output dictionary
    output = dict()
    output["acc"] = acc
    output["f1"] = f1
    output["conf_mat"] = conf_mat
    # output['roc'] = roc
    output["auc"] = auc

    return output
