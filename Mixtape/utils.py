"""Utility functions"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Bharath Ramsundar <bharath.ramsundar@gmail.com>
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

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import json
import numpy as np
from sklearn.utils import check_random_state
from sklearn.externals.joblib import load, dump
import sklearn.base, sklearn.pipeline
from numpy.linalg import norm

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

def verbosedump(value, fn, compress=1):
    """verbose wrapper around joblib.dump"""
    print('Saving "%s"... (%s)' % (fn, type(value)))
    dump(value, fn, compress=compress)

def verboseload(fn):
    """verbose wrapper around joblib.load"""
    print('loading "%s"...' % fn)
    return load(fn)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass


def categorical(pvals, size=None, random_state=None):
    """Return random integer from a categorical distribution

    Parameters
    ----------
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        should sum to 1.
    size : int or tuple of ints, optional
        Defines the shape of the returned array of random integers. If None
        (the default), returns a single float.
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    """
    cumsum = np.cumsum(pvals)
    if size is None:
        size = (1,)
        axis = 0
    elif isinstance(size, tuple):
        size = size + (1,)
        axis = len(size) - 1
    else:
        raise TypeError('size must be an int or tuple of ints')

    random_state = check_random_state(random_state)
    return np.sum(cumsum < random_state.random_sample(size), axis=axis)


def list_of_1d(y):
    if not hasattr(y, '__iter__') or len(y) == 0:
        raise ValueError('Bad input shape')
    if not hasattr(y[0], '__iter__'):
        return [np.array(y)]

    result = []
    for i, x in enumerate(y):
        value = np.array(x)
        if value.ndim != 1:
            raise ValueError(
                "Bad input shape. Element %d has shape %s, but "
                "should be 1D" % (i, str(value.shape)))
        result.append(value)
    return result

##########################################################################
# MSLDS Utils (experimental)
##########################################################################


def iter_vars(A, Q, N):
    """Utility function used to solve fixed point equation
       Q + A D A.T = D
       for D
     """
    V = np.eye(np.shape(A)[0])
    for i in range(N):
        V = Q + np.dot(A, np.dot(V, A.T))
    return V

##########################################################################
# END of MSLDS Utils (experimental)
##########################################################################

def map_drawn_samples(selected_pairs_by_state, trajectories):
    """Lookup trajectory frames using pairs of (trajectory, frame) indices.

    Parameters
    ----------
    selected_pairs_by_state : np.ndarray, dtype=int, shape=(n_states, n_samples, 2)
        selected_pairs_by_state[state, sample] gives the (trajectory, frame)
        index associated with a particular sample from that state.
    trajectories : list(md.Trajectory) or list(np.ndarray)
        The trajectories assocated with sequences,
        which will be used to extract coordinates of the state centers
        from the raw trajectory data.  This can also be a list of np.ndarray
        objects

    Returns
    -------
    frames_by_state : mdtraj.Trajectory
        Output will be a list of trajectories such that frames_by_state[state]
        is a trajectory drawn from `state` of length `n_samples`.  If trajectories
        are numpy arrays, the output will be numpy arrays instead of md.Trajectories
    
    Examples
    --------
    >>> selected_pairs_by_state = hmm.draw_samples(sequences, 3)
    >>> samples = map_drawn_samples(selected_pairs_by_state, trajectories)
    
    Notes
    -----
    YOU are responsible for ensuring that selected_pairs_by_state and 
    trajectories correspond to the same dataset!
    
    See Also
    --------
    utils.map_drawn_samples : Extract conformations from MD trajectories by index.
    ghmm.GaussianFusionHMM.draw_samples : Draw samples from GHMM    
    ghmm.GaussianFusionHMM.draw_centroids : Draw centroids from GHMM    
    """

    frames_by_state = []

    for state, pairs in enumerate(selected_pairs_by_state):
        frames = [trajectories[trj][frame] for trj, frame in pairs]
        try:  # If frames are mdtraj Trajectories
            state_trj = frames[0][0:0].join(frames)  # Get an empty trajectory with correct shape and call the join method on it to merge trajectories
        except AttributeError:
            state_trj = np.array(frames)  # Just a bunch of np arrays
        frames_by_state.append(state_trj)
    
    return frames_by_state

# TODO: FIX THIS!
def compute_eigenspectra(self):
    """
    Compute the eigenspectra of operators A_i
    """
    eigenspectra = np.zeros((self.n_states,
                            self.n_features, self.n_features))
    for k in range(self.n_states):
        eigenspectra[k] = np.diag(np.linalg.eigvals(self.As_[k]))
    return eigenspectra

# TODO: FIX THIS!
def load_from_json_dict(model, model_dict):
    # Check that the num of states and features agrees
    n_features = float(model_dict['n_features'])
    n_states = float(model_dict['n_states'])
    if n_features != self.n_features or n_states != self.n_states:
        raise ValueError('Invalid number of states or features')
    # read array values from the json dictionary
    Qs = []
    for Q in model_dict['Qs']:
        Qs.append(np.array(Q))
    As = []
    for A in model_dict['As']:
        As.append(np.array(A))
    bs = []
    for b in model_dict['bs']:
        bs.append(np.array(b))
    means = []
    for mean in model_dict['means']:
        means.append(np.array(mean))
    covars = []
    for covar in model_dict['covars']:
        covars.append(np.array(covar))
    # Transmat
    transmat = np.array(model_dict['transmat'])
    # Create the MSLDS model
    self.Qs_ = Qs
    self.As_ = As
    self.bs_ = bs
    self.means_ = means
    self.covars_ = covars
    self.transmat_ = transmat


class Subsampler(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Convert a list of feature time series (`X_all`) into a `lag_time` subsampled time series.

    Parameters
    ----------
    lag_time : int
        The lag time to subsample by
    sliding_window : bool, default=True
        If True, each time series is transformed into `lag_time` interlaced
        sliding-window (not statistically independent) sequences.  If 
        False, each time series is transformed into a single subsampled
        time series.
    """    
    def __init__(self, lag_time, sliding_window=True):
        self._lag_time = lag_time
        self._sliding_window = sliding_window

    def fit(self, X_all, y=None):
        return self

    def transform(self, X_all, y=None):
        """Subsample several time series.

        Parameters
        ----------
        X_all : list(np.ndarray)
            List of feature time series

        Returns
        -------
        features : list(np.ndarray), length = len(X_all)
            The subsampled trajectories.
        """
        if self._sliding_window:
            return [X[k::self._lag_time] for k in range(self._lag_time) for X in X_all]
        else:
            return [X[::self._lag_time] for X in X_all]
