import os
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

from src.utils import get_project_root


def kfold_cv_train_grid(
    data,
    grid_penalizer,
    grid_l1_ratio,
    k,
    scoring_method="log_likelihood",
    seed=8971,
    verbose=False,
    fitter_kwargs=None,
):
    """
    Performs a K-fold cross-validation of the data using a CoxPHFitter over the
    specified grid of penalizer and l1_ratio points.
    """
    mg = np.meshgrid(grid_penalizer, grid_l1_ratio)
    grid_points = np.vstack((mg[0].flatten(), mg[1].flatten())).T
    print(f"Training {k=} folds over {len(grid_points)} grid points.")

    fitters = []
    for penalizer, l1_ratio in grid_points:
        if verbose:
            print(f"(penalizer, l1_ratio) = ({penalizer}, {l1_ratio})")
        fitters.append(CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio))

    scores = k_fold_cross_validation(
        fitters,
        data,
        duration_col="Survival months",
        event_col="Event indicator",
        k=k,
        scoring_method=scoring_method,
        seed=seed,
        fitter_kwargs=fitter_kwargs,
    )
    scores = np.array(scores)
    results = pd.DataFrame.from_dict(
        {
            "penalizer": grid_points[:, 0],
            "l1_ratio": grid_points[:, 1],
            "score0": scores[:, 0],
            "score1": scores[:, 1],
            "score2": scores[:, 2],
            "score_mean": scores.mean(axis=1),
        }
    )
    return results


def get_best_params_from_kfold_results(
    results: pd.DataFrame,
    print_results: bool = True,
) -> pd.Series:
    """
    Returns the best hyperparameters from the kfold_cv_train_grid results.
    """
    params_best = results.iloc[results["score_mean"].idxmax()]
    if print_results:
        print("\nBest params:")
        print(f"  penalizer:  {params_best['penalizer']}")
        print(f"  l1_ratio:   {params_best['l1_ratio']}")
        print(f"  score_mean: {params_best['score_mean']}")
    return params_best


def save_gridsearch_results(
    results: pd.DataFrame,
    filename="gridsearch.csv",
    overwrite=False,
) -> None:
    """
    Save the grid search results to a file.

    Parameters
    ----------
    results : pd.DataFrame
        The grid search results.
    filename : str, optional
        The filename to save the results to. Default is 'gridsearch.csv'.
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
    results.to_csv(filepath, index=False)


def load_gridsearch_results(
    filename="gridsearch.csv",
) -> pd.DataFrame:
    """
    Load the grid search results from a file.

    Parameters
    ----------
    filename : str, optional
        The filename to load the results from. Default is 'gridsearch.csv'.

    Returns
    -------
    results : pd.DataFrame
        The grid search results.
    """
    filepath = os.path.join(get_project_root(), "models", filename)
    results = pd.read_csv(filepath)
    return results
