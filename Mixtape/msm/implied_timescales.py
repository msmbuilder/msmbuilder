from .. import param_sweep
from . import MarkovStateModel
import numpy as np

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

    Returns
    -------
    timescales : np.ndarray, shape = [n_models, n_timescales]
        The slowest timescales (in units of lag times) for each
        model.
    """

    if msm is None:
        msm = MarkovStateModel()

    param_grid = {'lag_time' : lag_times}

    models = param_sweep(msm, sequences, param_grid, n_jobs=n_jobs)

    timescales = np.array([model.timescales_[:n_timescales] for model in models])

    return timescales
