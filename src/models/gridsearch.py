import os
import pickle
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.utils import get_project_root
from src.models.io import save_model


def build_2d_gridsearch(
    estimator: BaseEstimator,
    cv,
    l1_ratios,
    alphas,
    n_jobs=-1,
    verbose=0,
) -> GridSearchCV:
    """
    Build a 2D grid search over the hyperparameters of the Coxnet model.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The Coxnet model to optimize.
    cv : sklearn.model_selection.KFold
        The K-fold cross-validation object.
    l1_ratios : np.ndarray
        The l1 ratios to search over.
    alphas : np.ndarray
        The alphas to search over.
    n_jobs : int, optional
        The number of parallel jobs to run for the grid search.
        Default is -1, which uses all available CPUs.
    verbose : int, optional
        The verbosity level of the grid search. Default is 0.

    Returns
    -------
    cv_search : sklearn.model_selection.GridSearchCV
        The grid search object.
    """
    # argument alphas of CoxnetSurvivalAnalysis must be an array at each grid point
    wrapped_alphas = [[alpha] for alpha in alphas]
    cv_search_dist = {
        "l1_ratio": l1_ratios,
        "alphas": wrapped_alphas,
    }
    cv_search = GridSearchCV(
        estimator,
        cv_search_dist,
        cv=cv,
        error_score=0.5,  # replaces fit errors with random models
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return cv_search


def build_1d_gridsearch(
    estimator: BaseEstimator,
    cv,
    alphas,
    n_jobs=-1,
    verbose=0,
) -> GridSearchCV:
    """
    Build a 1D grid search over the hyperparameters of the Coxnet model.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The Coxnet model to optimize.
    cv : sklearn.model_selection.KFold
        The K-fold cross-validation object.
    alphas : np.ndarray
        The alphas to search over.
    n_jobs : int, optional
        The number of parallel jobs to run for the grid search.
        Default is -1, which uses all available CPUs.
    verbose : int, optional
        The verbosity level of the grid search. Default is 0.

    Returns
    -------
    cv_search : sklearn.model_selection.GridSearchCV
        The grid search object.
    """
    # argument alphas of CoxnetSurvivalAnalysis must be an array at each grid point
    wrapped_alphas = [[alpha] for alpha in alphas]
    cv_search_dist = {"alphas": wrapped_alphas}
    cv_search = GridSearchCV(
        estimator,
        cv_search_dist,
        cv=cv,
        error_score=0.5,  # replaces fit errors with random models
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return cv_search


def run_2d_gridsearch(
    estimator: BaseEstimator,
    X_transformed: np.ndarray,
    y_train: pd.Series,
    l1_ratios: np.ndarray,
    alphas: np.ndarray,
    n_splits: int,
    n_jobs=-1,
    verbose=0,
    max_iter=10000,
    overwrite=False,
    save_best_model=False,
    save_results=False,
    filename_model="coxnet_model_grid_search_2d.pkl",
    filename_results="coxnet_2d_grid_search.csv",
) -> pd.DataFrame | dict:
    """
    Perform a 2D grid search over the hyperparameters of the Coxnet model.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The Coxnet model to optimize.
    X_transformed : np.ndarray
        The transformed feature matrix.
    y_train : pd.Series
        The target variable.
    l1_ratios : np.ndarray
        The l1 ratios to search over.
    alphas : np.ndarray
        The alphas to search over.
    n_splits : int
        The number of splits for the K-fold cross-validation.
    n_jobs : int, optional
        The number of parallel jobs to run for the grid search.
        Default is -1, which uses all available CPUs.
    verbose : int, optional
        The verbosity level of the grid search. Default is 0.
    max_iter : int, optional
        The maximum number of iterations used when fitting the Coxnet model. Default is 10000.
    overwrite : bool, optional
        Whether to overwrite the existing model and results files. Default is False.
    save_best_model : bool, optional
        Whether to save the best model to a file. Default is False.
    save_results : bool, optional
        Whether to save the grid search results to a file. Default is False.
    filename_model : str, optional
        The filename to save the best model to. Default is 'coxnet_model_grid_search_2d.pkl'.
    filename_results : str, optional
        The filename to save the grid search results to. Default is 'coxnet_2d_grid_search.csv'.

    Returns
    -------
    cv_results : pd.DataFrame
        The grid search results.
    best_params : dict
        The best hyperparameters found by the grid search.
    """
    # Make the K-fold split deterministic for now
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=2984)

    # Build the grid search metaestimator and fit it
    cv_search = build_2d_gridsearch(estimator, cv, l1_ratios, alphas, n_jobs, verbose)
    cv_search.fit(X_transformed, y_train.to_records(index=False))

    # Get a results summary and best parameters from the grid search
    cv_results = pd.DataFrame(cv_search.cv_results_)
    best_params = cv_search.best_params_

    if save_best_model:
        save_model(cv_search.best_estimator_, filename_model, overwrite)
    if save_results:
        save_gridsearch_results(cv_search, filename_results, overwrite)

    return cv_results, best_params


def run_1d_gridsearch(
    estimator: BaseEstimator,
    X_transformed: np.ndarray,
    y_train: pd.Series,
    l1_ratio: float,
    alphas: np.ndarray,
    n_splits: int,
    n_jobs=-1,
    verbose=0,
    max_iter=10000,
    overwrite=False,
    save_best_model=False,
    save_results=True,
    filename_model="coxnet_model.pkl",
    filename_results="coxnet_1d_grid_search.csv",
) -> pd.DataFrame | dict:
    """
    Perform a 1D grid search over the hyperparameters of the Coxnet model.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The Coxnet model to optimize.
    X_transformed : np.ndarray
        The transformed feature matrix.
    y_train : pd.Series
        The target variable.
    l1_ratio : float
        The fixed l1 ratio to use.
    alphas : np.ndarray
        The alphas to search over.
    n_splits : int
        The number of splits for the K-fold cross-validation.
    n_jobs : int, optional
        The number of parallel jobs to run for the grid search.
        Default is -1, which uses all available CPUs.
    verbose : int, optional
        The verbosity level of the grid search. Default is 0.
    max_iter : int, optional
        The maximum number of iterations used when fitting the Coxnet model. Default is 10000.
    overwrite : bool, optional
        Whether to overwrite the existing model and results files. Default is False.
    save_best_model : bool, optional
        Whether to save the best model to a file. Default is False.
    save_results : bool, optional
        Whether to save the grid search results to a file. Default is False.
    filename_model : str, optional
        The filename to save the best model to. Default is 'coxnet_model.pkl'.
    filename_results : str, optional
        The filename to save the grid search results to. Default is 'coxnet_1D_grid_search.csv'.

    Returns
    -------
    cv_results : pd.DataFrame
        The grid search results.
    best_params : dict
        The best hyperparameters found by the grid search.
    """
    # Make the K-fold split deterministic for now
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=2984)

    # Build the grid search metaestimator and fit it
    cv_search = build_1d_gridsearch(estimator, cv, alphas, n_jobs, verbose)
    cv_search.fit(X_transformed, y_train.to_records(index=False))

    # Get a results summary and best parameters from the grid search
    cv_results = pd.DataFrame(cv_search.cv_results_)
    best_params = cv_search.best_params_

    if save_best_model:
        save_model(cv_search.best_estimator_, filename_model, overwrite)
    if save_results:
        save_gridsearch_results(cv_search, filename_results, overwrite)

    return cv_results, best_params


def save_gridsearch_results(
    cv_search: GridSearchCV,
    filename="coxnet_grid_search.csv",
    overwrite=False,
) -> None:
    """
    Save the grid search results to a file.

    Parameters
    ----------
    cv_search : sklearn.model_selection.GridSearchCV
        The grid search object.
    filename : str, optional
        The filename to save the results to. Default is 'coxnet_grid_search.csv'.
    overwrite : bool, optional
        Whether to overwrite the existing file. Default is False.

    Returns
    -------
    None
    """
    filepath = os.path.join(get_project_root(), "models", filename)
    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(
            f"File {filepath} already exists! Set overwrite=True to overwrite."
        )
    print(f"Saving grid search results to {filepath}")
    cv_results = pd.DataFrame(cv_search.cv_results_)
    cv_results.to_csv(filepath, index=False)


def load_gridsearch_results(
    filename="coxnet_grid_search.csv",
) -> pd.DataFrame:
    """
    Load the grid search results from a file.

    Parameters
    ----------
    filename : str, optional
        The filename to load the results from. Default is 'coxnet_grid_search.csv'.

    Returns
    -------
    cv_results : pd.DataFrame
        The grid search results.
    best_params : dict
        The best hyperparameters found by the grid search.
    """
    print(f"Loading grid search results from {filename}")
    filepath = os.path.join(get_project_root(), "models", filename)
    cv_results = pd.read_csv(filepath)
    # Determine best parameters
    best_idx = cv_results["rank_test_score"].idxmin()
    best_params = cv_results.loc[best_idx, "params"]
    best_params = eval(best_params)
    return cv_results, best_params
