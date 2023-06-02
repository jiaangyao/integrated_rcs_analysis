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

import utils.torch_utils as ptu
from biomarker.training.model_infrastructure import get_model, BaseModel
from biomarker.training.torch_model_infrastructure import get_model_ray
from biomarker.training.get_model_params import get_model_params
from biomarker.training.correct_data_dim import correct_data_dim
from utils.combine_labels import combine_labels


def test_if_torch_model(str_model):
    return get_model_params(str_model, 1, 1)[2]


def train_model(
    vec_features: list[npt.NDArray],
    vec_y_class: list[npt.NDArray],
    y_stim_test: npt.NDArray,
    n_class: int = 4,
    str_model: str = "LDA",
    hashmap: dict[str, int] | None = None,
    bool_use_ray: bool = False,
    bool_use_gpu: bool = False,
    n_cpu_per_process: int | float = 1,
    n_gpu_per_process: int | float = 0,
) -> dict[str, npt.NDArray | int | float | None]:
    # obtain the dimensionality
    n_input = vec_features[0].shape[1]
    n_class_model = len(np.unique(vec_y_class[0]))

    # now define the model parameters and correct the feature dimensionality
    model_params, train_params, bool_torch = get_model_params(
        str_model, n_input, n_class_model
    )
    vec_features = correct_data_dim(str_model, vec_features)

    # obtain the parameters for training and append it to the configuraiton

    # now define the model
    if bool_torch:
        # unpack the features first
        features_train, features_valid, features_test = vec_features
        y_class_train, y_class_valid, y_class_test = vec_y_class

        # now start the model training
        # initialize GPU if not using ray since not initialized elsewhere
        if not bool_use_ray and bool_use_gpu:
            ptu.init_gpu(bool_use_gpu, bool_use_best_gpu=True)

        model = get_model_ray(
            str_model,
            model_params,
            bool_use_ray=bool_use_ray,
            bool_use_gpu=bool_use_gpu,
            n_cpu_per_process=n_cpu_per_process,
            n_gpu_per_process=n_gpu_per_process,
        )

        # train the model
        if bool_use_ray and bool_use_gpu:
            model.train.remote(
                features_train,
                y_class_train,
                valid_data=features_valid,
                valid_label=y_class_valid,
                **train_params
            )
        else:
            model.train(
                features_train,
                y_class_train,
                valid_data=features_valid,
                valid_label=y_class_valid,
                **train_params
            )

        # next generate the predictions
        if bool_use_ray and bool_use_gpu:
            y_class_pred = model.predict.remote(features_test)
        else:
            y_class_pred = model.predict(features_test)
        if np.ndim(y_class_pred) > 1:
            y_class_pred = np.argmax(y_class_pred, axis=1)
        
        # estimate the ROC curve
        if bool_use_ray and bool_use_gpu:
            y_class_pred_scores = model.predict_proba.remote(features_test)
        else:
            y_class_pred_scores = model.predict_proba(features_test)
        
        # obtain the object from ray remote if using gpu
        if bool_use_ray and bool_use_gpu:
            [y_class_pred, y_class_pred_scores] = ray.get([y_class_pred, y_class_pred_scores])
            
    else:
        # warn if using GPU for sklearn
        if bool_use_gpu:
            Warning("Using GPU for sklearn model training")

        # unpack the features first
        features_train, features_test = vec_features
        y_class_train, y_class_test = vec_y_class

        # now obtain the models
        model = get_model(str_model, model_params)
        model.train(features_train, y_class_train)

        # next generate the predictions
        y_class_pred = model.predict(features_test)
        if y_class_pred.ndim > 1:
            y_class_pred = np.argmax(y_class_pred, axis=1)
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
    roc = roc_curve(y_class_test, y_class_pred_scores)
    auc = roc_auc_score(y_class_test, y_class_pred_scores)

    # create the output dictionary
    output = dict()
    output["acc"] = acc
    output["f1"] = f1
    output["conf_mat"] = conf_mat
    # output['roc'] = roc
    output["auc"] = auc

    return output
