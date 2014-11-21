# Author: Christian Schwantes <schwancr@stanford.edu>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import
from mdtraj.utils.six.moves import xrange

from sklearn import clone
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

def param_sweep(model, sequences, param_grid, n_jobs=1):
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
