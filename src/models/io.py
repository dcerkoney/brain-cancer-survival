import os
import pickle

from lifelines import CoxPHFitter

from src.utils import get_project_root


def save_model(
    fitter: CoxPHFitter,
    filename="model.pkl",
    overwrite=False,
) -> None:
    """
    Serialize a model to a file using pickle.

    Parameters
    ----------
    fitter : lifelines.CoxPHFitter
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
        pickle.dump(fitter, f)


def load_model(
    filename="model.pkl",
) -> CoxPHFitter:
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
        if not isinstance(model, CoxPHFitter):
            raise ValueError("Invalid model: not a lifelines.CoxPHFitter!")
        return model
