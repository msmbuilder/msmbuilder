
from __future__ import print_function, division, absolute_import
from mdtraj.utils.six.moves import xrange

from sklearn.grid_search import ParameterGrid
import numpy as np
from joblib import Parallel, delayed

def _fit_helper(args):
    """
    helper for fitting many models on some data
    """
    model, sequences = args
    model.fit(sequences)

    return model

def fit_many(model, sequences, param_grid, n_jobs=1):
    """
    Helper to fit many models to the same data

    Parameters
    ----------
    model : mixtape.BaseEstimator
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
    iter_args = ((clone(model).set_params(params), sequences) for params in param_grid)

    models = Parallel(n_jobs=n_jobs)(delayed(_fit_helper)((args) for args in iter_args))

    return models

def implied_timescales(sequences, lag_times, n_timescales=10, 
    msm=None, n_jobs=1):
    """
    Calculate the implied timescales for a given MSM.

    Parameters
    ----------
    sequences : list of array-like
        List of sequences, or a single sequence. Each 
        sequence should be a 1D iterable of state
        labels. Labels can be integers, strings, or
        other orderable objects.
    lag_times : array-like
        Lag times to calculate implied timescales at.
    n_timescales : int, optional
        Number of timescales to calculate.
    msm : mixtape.MarkovStateModel, optional
        Instance of an MSM to specify parameters other
        than the lag time. If None, then the default
        parameters (as implemented by mixtape.MarkovStateModel)
        will be used.
    n_jobs : int, optional
        Number of jobs to run in parallel
    """

    if msm is None:
        msm = MarkovStateModel()

    param_grid = {'lag_time' : lag_times}

    models = fit_many(msm, sequences, param_grid, n_jobs=n_jobs)

    timescales = [model.timescales_[:n_timescales] for model in models]

    return timescales
