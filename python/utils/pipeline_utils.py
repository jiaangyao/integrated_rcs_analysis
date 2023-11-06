from sklearn.pipeline import Pipeline
import os
import importlib
import duckdb
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
import inspect
import types
import pkgutil


def get_callable_function(func_name):
    """
    Get a callable function handle from a string like 'module.function'.

    Parameters:
    func_name (str): The name of the function to get.

    Returns:
    callable: A callable function handle for the input function name, or None if the function cannot be found.
    """
    try:
        module_name, func_name = func_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        if callable(func):
            return func
    except (ValueError, AttributeError, ModuleNotFoundError):
        pass
    return None


def convert_string_to_callable(libs: list[object], func: str) -> callable:
    """
    Converts a string to a callable function.

    Args:
        libs (list[object]): A list of objects (i.e. imported libraries) to search for the callable function.
        func (callable): A callable function.

    Returns:
        callable: A callable function.
    """
    if isinstance(func, str):
        # First check if the func string is the entire module path
        attempt_1 = get_callable_function(func)
        if attempt_1 is not None:
            return attempt_1
        else:
            # Next, check if the func string is just the function name, and search the libs for the function
            for obj in libs:
                if hasattr(obj, func):
                    return getattr(obj, func)
    else:
        return func


def create_instance_from_directory(directory, class_name, *args, **kwargs):
    """
    Search a directory and its subdirectories for a class by name and create an instance of that class.

    Parameters:
    directory (str): The path to the directory to search.
    class_name (str): The name of the class to create an instance of.
    *args: Positional arguments to pass to the class constructor.
    **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
    object: An instance of the class.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                module_name = os.path.splitext(file)[0]
                module_path = os.path.join(root, file)
                try:
                    module = importlib.import_module(module_path)
                except ModuleNotFoundError:
                    continue
                for _, cls in module.__dict__.items():
                    if isinstance(cls, type) and cls.__name__ == class_name:
                        return cls(*args, **kwargs)
    raise ValueError(f"Class {class_name} not found in directory {directory}")


def create_instance_from_module(module_name, class_name, *args, **kwargs):
    """
    Search a module and its submodules for a class by name and create an instance of that class.

    Parameters:
    module_name (str): The name of the module to search.
    class_name (str): The name of the class to create an instance of.
    *args: Positional arguments to pass to the class constructor.
    **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
    object: An instance of the class.
    """
    module = importlib.import_module(module_name)
    for _, cls in module.__dict__.items():
        if isinstance(cls, type) and cls.__name__ == class_name:
            return cls(*args, **kwargs)
    for name in dir(module):
        submodule = getattr(module, name)
        if isinstance(submodule, types.ModuleType):
            try:
                instance = create_instance_from_module(
                    submodule.__name__, class_name, *args, **kwargs
                )
                return instance
            except ValueError:
                continue
    raise ValueError(f"Class {class_name} not found in module {module_name}")


def find_and_load_class(module_name, class_name, args=[], kwargs={}):
    # Import the module.
    try:
        module = importlib.import_module(module_name)
    # TODO: Handle ImportErrors differently instead of excepting them,
    # as the desired class may be in a module that yields import error
    except (ModuleNotFoundError, ImportError) as e:
        # If module is not found, just return None
        if isinstance(e, ModuleNotFoundError): print(f"Module {module_name} not found, skipping")
        if isinstance(e, ImportError): print(f"Module {module_name} could not be imported, skipping")
        return None
    

    # Check if the class is in the current module.
    cls = getattr(module, class_name, None)
    # Return an instance of the class.
    if cls:
        if kwargs and args:
            return cls(*args, **kwargs)
        elif kwargs:
            return cls(**kwargs)
        elif args:
            return cls(*args)
        else:
            return cls()

    # If the module has submodules, search them recursively.
    if hasattr(module, "__path__"):
        for _, modname, ispkg in pkgutil.iter_modules(module.__path__):
            # Recursively call the function for each submodule.
            instance = find_and_load_class(
                f"{module_name}.{modname}", class_name, args, kwargs
            )
            if instance:
                return instance  # Return the instance if found.


def create_transformer(func, **kwargs):
    """
    Create a scikit-learn compatible transformer from an arbitrary function.

    Parameters:
    func (callable): The function to transform the input data.

    Returns:
    transformer (TransformerMixin): A scikit-learn compatible transformer object.
    """

    class Transformer(BaseEstimator, TransformerMixin):
        def __init__(self, func, **kwargs):
            self.func = func
            self.kwargs = kwargs

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return self.func(X, **self.kwargs)

    transformer = Transformer(func, **kwargs)
    return transformer


def create_transform_pipeline(steps):
    """
    Create a scikit-learn Pipeline object from a dictionary of function names, handles, and arguments.

    Parameters:
    steps (dict): A dictionary of function names, handles, and arguments.

    Returns:
    pipe (Pipeline): A scikit-learn Pipeline object that pipes together the dictionary entries.
    """
    pipe_steps = []
    pipe_step_paths = []
    for name, (func, kwargs) in steps.items():
        if kwargs is None:
            kwargs = {}
        # Retrieve the module path of the function for logging
        module_name = inspect.getmodule(func).__name__.rsplit(".", 1)[0]
        if module_name in name:
            pipe_step_paths.append((name, kwargs))
        else:
            pipe_step_paths.append((module_name + f".{name}", kwargs))

        # Check to see if the function is a TransformerMixin object,
        # so that it may be incorporated into the Sklearn.Pipeline
        if not isinstance(func, TransformerMixin):
            transformer_func = create_transformer(func, **kwargs)

        pipe_steps.append((name, transformer_func))

    pipe = Pipeline(pipe_steps)
    return pipe, pipe_step_paths
