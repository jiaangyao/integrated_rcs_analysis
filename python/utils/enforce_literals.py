from typing import Literal, get_args, get_origin
from sys import _getframe


def enforce_literals(function):
    """Enforces that arguments are in the specified Literal options

    Args:
        function (func.): _description_

    Raises:
        ValueError: raised if the argument is not in the Literal options
    """
    kwargs = _getframe(1).f_locals
    for name, type_ in function.__annotations__.items():
        value = kwargs.get(name)
        options = get_args(type_)
        if get_origin(type_) is Literal and name in kwargs and value not in options:
            raise ValueError(f"'{value}' is not in {options} for '{name}'")
