from types import MappingProxyType

from .discriminant_analysis import LDAModel, QDAModel
from .support_vector_machine import SVMModel
from .gaussian_process import GPModel
from .tree_based_model import RandomForestModel
from .boosting_method import AdaBoostModel, GradientBoostingModel


def init_model_sklearn(
    str_model: str,
    model_args: list | tuple = tuple(),
    model_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
    ensemble_args: list | tuple = tuple(),
    ensemble_kwargs: dict | MappingProxyType = MappingProxyType(dict()),
):

    # parse the arguments and initialize the corresponding model
    if str_model == "LDA":
        model = LDAModel(model_args, model_kwargs)
    elif str_model == "QDA":
        model = QDAModel(model_args, model_kwargs)
    elif str_model == "SVM":
        model = SVMModel(model_args, model_kwargs)
    elif str_model == "RF":
        model = RandomForestModel(model_args, model_kwargs)
    elif str_model == "GP":
        model = GPModel(model_args, model_kwargs)
    elif str_model == "AdaBoost":
        model = AdaBoostModel(model_args, model_kwargs, ensemble_args, ensemble_kwargs)
    elif str_model == "GBoost":
        model = GradientBoostingModel(model_args, model_kwargs)
    else:
        raise NotImplementedError("f{str_model} is not implemented yet.")

    return model
