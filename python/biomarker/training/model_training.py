import ray
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

from utils.combine_labels import combine_labels


def train_model(
    model,
    vec_features: list[npt.NDArray],
    vec_y_class: list[npt.NDArray],
    y_stim_test: npt.NDArray,
    trainer_cfg: DictConfig | dict,
    n_class: int = 4,
    hashmap: dict[str, int] | None = None,
    # bool_return_pred: bool = False,
) -> dict[str, npt.NDArray | int | float | None]:
    # now train the model
    # if using pytorch models
    if model.bool_torch:
        # unpack the features first
        features_train, features_valid, features_test = vec_features
        y_class_train, y_class_valid, y_class_test = vec_y_class
        
        # if asked to normalize the features
        # TODO: integrate this into the DataLoader class in torch
        if trainer_cfg['bool_whiten_feature']:
            # calculate mean and std from training set
            mean_train = np.mean(features_train, axis=0)
            sigma_train = np.std(features_train, axis=0)
            
            # apply normalization to all sets
            features_train = (features_train - mean_train) / sigma_train
            features_valid = (features_valid - mean_train) / sigma_train
            features_test = (features_test - mean_train) / sigma_train
            
        # optionally perform L2 normalization
        # TODO: verify this function
        if trainer_cfg['bool_l2_normalize_feature']:
            # calculate l2 norm from training set
            l2_norm_train = np.linalg.norm(features_train, axis=1)
            
            # apply normalization to all sets
            features_train = features_train / l2_norm_train[:, None]
            features_valid = features_valid / l2_norm_train[:, None]
            features_test = features_test / l2_norm_train[:, None]
        
        # TODO: get rid of following line, trainer_cfg should not be a dict
        if trainer_cfg is not dict:
            trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)
            trainer_cfg.pop('bool_whiten_feature')
            trainer_cfg.pop('bool_l2_normalize_feature')
        
        # train the model
        model.train(
            features_train,
            y_class_train,
            valid_data=features_valid,
            valid_label=y_class_valid,
            **trainer_cfg
        )

    # if using sklearn models
    else:
        # unpack the features first
        features_train, features_test = vec_features
        y_class_train, y_class_test = vec_y_class

        # now obtain the models
        model.train(features_train, y_class_train)

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
    
    # # optionally return predictions as well
    # if bool_return_pred:
    #     output['pred'] = dict()
    #     output['pred']['y_class_test'] = y_class_test
    #     output['pred']['y_class_pred_scores'] = y_class_pred_scores
        
    #     output['pred']['full_label_test'] = full_label_test
    #     output['pred']['full_label_pred'] = full_label_pred

    return output
