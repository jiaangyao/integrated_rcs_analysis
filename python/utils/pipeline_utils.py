from sklearn.pipeline import Pipeline
import os
import importlib
import duckdb
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(data_params):
    """
    Load data from a file or database.
    """
    if data_params['source'] == 'database':
        con = duckdb.connect(data_params['database_path'], read_only=True)
        return con.sql(data_params['query']).pl()
    else:
        return pl.read_parquet(data_params['data_path'])
        
        
def convert_string_to_callable(objs: list[object], func: str) -> callable:
    """
    TODO: Move to a utils file
    Converts a string to a callable function.
    
    Args:
        func (callable): A callable function.
    
    Returns:
        callable: A callable function.
    """
    if isinstance(func, str):
        for obj in objs:
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
            if file.endswith('.py'):
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
    for submodule_name in module.__all__:
        submodule = importlib.import_module(f"{module_name}.{submodule_name}")
        try:
            instance = create_instance_from_module(submodule.__name__, class_name, *args, **kwargs)
            return instance
        except ValueError:
            continue
    raise ValueError(f"Class {class_name} not found in module {module_name}")


def FunctionTransformer(func):
    """
    Create a scikit-learn compatible transformer from an arbitrary function.
    
    Parameters:
    func (callable): The function to transform the input data.
    
    Returns:
    transformer (TransformerMixin): A scikit-learn compatible transformer object.
    """
    class Transformer(BaseEstimator, TransformerMixin):
        def __init__(self, func):
            self.func = func
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return self.func(X)
    
    transformer = Transformer(func)
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
    for name, (func, args) in steps.items():
        # Check to see if the function is a TransformerMixin object, 
        # so that it may be incorporated into the Sklearn.Pipeline
        if not isinstance(func, TransformerMixin):
            func = FunctionTransformer(func)
            
        pipe_steps.append((name, func(**args)))
        
    pipe = Pipeline(pipe_steps)
    return pipe