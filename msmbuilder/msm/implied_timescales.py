# Author: Christian Schwantes
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.


import numpy as np
from ..utils import param_sweep
from . import MarkovStateModel


def implied_timescales(sequences, lag_times, n_timescales=10,
                       msm=None, n_jobs=1, verbose=0):
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
    msm : msmbuilder.msm.MarkovStateModel, optional
        Instance of an MSM to specify parameters other
        than the lag time. If None, then the default
        parameters (as implemented by msmbuilder.msm.MarkovStateModel)
        will be used.
    n_jobs : int, optional
        Number of jobs to run in parallel

    Returns
    -------
    timescales : np.ndarray, shape = [n_models, n_timescales]
        The slowest timescales (in units of lag times) for each
        model.
    """

    if msm is None:
        msm = MarkovStateModel()

    param_grid = {'lag_time' : lag_times}
    models = param_sweep(msm, sequences, param_grid, n_jobs=n_jobs,
                         verbose=verbose)
    timescales = [m.timescales_ for m in models]
    n_timescales = min(n_timescales, min(len(ts) for ts in timescales))
    timescales = np.array([ts[:n_timescales] for ts in timescales])
    return timescales
