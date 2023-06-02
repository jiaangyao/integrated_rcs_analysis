import typing as tp
import numpy as np
import numpy.typing as npt


def get_model_params(str_model, n_input, n_class):
    # TODO: finish the documentation
    """_summary_

    Args:
        str_model (_type_): _description_
        n_input (_type_): _description_
        n_class (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create the empty dictionary holding the output
    bool_torch = False
    train_params = dict[str, tp.Any]()

    # next test against the different model types to find proper parameters
    if str_model == "LDA":
        model_params = {}
    elif str_model == "QDA":
        model_params = {"tol": 1e-9}
    elif str_model == "SVM":
        model_params = {"probability": True}
    elif str_model == "RF":
        # model_params = {'max_depth': 3, 'n_estimators': 300}
        model_params = {"n_estimators": 300}
    elif str_model == "MLP":
        model_params = {
            "args": [n_input, n_class],
            "kwargs": {
                "n_layer": 3,
                "dropout": 0.35,
                "str_act": "leaky_relu",
                "lr": 5e-4,
                "lam": 1e-5,
                "hidden_size": 128,
            },
        }
        train_params = {"n_epoch": 100, "batch_size": 128, "bool_verbose": False}
        bool_torch = True
    elif str_model == "RNN":
        model_params = {
            "args": [n_input, n_class],
            "kwargs": {
                "bool_cnn": False,
                "cnn_act_func": "identity",
                "n_rnn_layer": 2,
                "rnn_dim": 16,
                "rnn_dropout": 0.6,
                "final_dropout": 0.4,
                "lr": 5e-4,
                "lam": 0,
            },
        }
        train_params = {"n_epoch": 100, "batch_size": 128, "bool_verbose": False}
        bool_torch = True
    else:
        model_params = {}

    return model_params, train_params, bool_torch
