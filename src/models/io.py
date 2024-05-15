import os
import pickle

from src.utils import get_project_root
from sklearn.base import BaseEstimator


def save_model(
    estimator: BaseEstimator,
    filename="model.pkl",
    overwrite=False,
) -> None:
    """
    Serialize a model to a file using pickle.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The model to save.
    filename : str, optional
        The filename to save the model to. Default is 'model.pkl'.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. Default is False.

    Returns
    -------
    None
    """
    filepath = os.path.join(get_project_root(), "models", filename)
    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(
            f"File {filepath} already exists! Set overwrite=True to overwrite."
        )
    with open(filepath, "wb") as f:
        pickle.dump(estimator, f)


def load_model(
    filename="model.pkl",
) -> BaseEstimator:
    """
    De-serialize a model from a file using pickle.

    Parameters
    ----------
    filename : str, optional
        The filename to load the model from. Default is 'model.pkl'.

    Returns
    -------
    object
        The de-serialized model object.
    """
    filepath = os.path.join(get_project_root(), "models", filename)
    with open(filepath, "rb") as f:
        model = pickle.load(f)
        if not isinstance(model, BaseEstimator):
            raise ValueError("Invalid model: not a sklearn.base.BaseEstimator!")
        return model
