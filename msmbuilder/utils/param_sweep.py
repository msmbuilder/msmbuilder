from __future__ import print_function, division, absolute_import
from sklearn import clone
from sklearn.grid_search import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed

__all__ = ['param_sweep']


def param_sweep(model, sequences, param_grid, n_jobs=1, verbose=0):
    """Fit a series of models over a range of parameters.

    Parameters
    ----------
    model : msmbuilder.BaseEstimator
        An *instance* of an estimator to be used
        to fit data.
    sequences : list of array-like
        List of sequences, or a single sequence. Each
        sequence should be a 1D iterable of state
        labels. Labels can be integers, strings, or
        other orderable objects.
    param_grid : dict or sklearn.grid_search.ParameterGrid
        Parameter grid to specify models to fit. See
        sklearn.grid_search.ParameterGrid for an explanation
    n_jobs : int, optional
        Number of jobs to run in parallel using joblib.Parallel

    Returns
    -------
    models : list
        List of models fit to the data according to
        param_grid
    """

    if isinstance(param_grid, dict):
        param_grid = ParameterGrid(param_grid)
    elif not isinstance(param_grid, ParameterGrid):
        raise ValueError("param_grid must be a dict or ParamaterGrid instance")

    # iterable with (model, sequence) as items
    iter_args = ((clone(model).set_params(**params), sequences)
                 for params in param_grid)

    models = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_param_sweep_helper)(args) for args in iter_args)

    return models


def _param_sweep_helper(args):
    """
    helper for fitting many models on some data
    """
    model, sequences = args
    model.fit(sequences)

    return model
