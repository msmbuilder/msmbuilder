# Author: Robert McGibbon <rmcgibbo@gmail.com>
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

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import

import time
import warnings
import numpy as np
import scipy.sparse
import scipy.linalg
from sklearn.utils import column_or_1d
from sklearn.base import BaseEstimator
from mdtraj.utils import ensure_type
from mixtape._markovstatemodel import _transmat_mle_prinz

__all__ = ['MarkovStateModel']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class MarkovStateModel(BaseEstimator):
    """Reversible Markov State Model

    Parameters
    ----------
    lag_time : int
        The lag time of the model
    n_timescales : int, optional
        The number of dynamical timescales to calculate when diagonalizing
        the transition matrix. By default, the maximum number will be
        calculated, which, for ARPACK, is n_states - 3.
    reversible_type : {'mle', 'transpose', None}
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization (BFGS), and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    ergodic_trim : bool
        Trim states to achieve an ergodic model. The model is restricted
        to the largest strongly connected component in the undirected
        transition counts.
    prior_counts : float, optional
        Add a number of "pseudo counts" to each entry in the counts matrix,
        `rawcounts_`. When prior_counts == 0 (default), the assigned transition
        probability between two states with no observed transitions will be zero,
        whereas when prior_counts > 0, even this unobserved transitions will be
        given nonzero probability. Note that prior_counts _totally_ destroys
        performance when the number of states is large, because none of the
        matrices are sparse anymore.
    verbose : bool
        Enable verbose printout

    Attributes
    ----------
    mapping_ : dict
        Mapping between "input" labels and internal state indices used by the
        counts and transition matrix for this Markov state model. Input states
        need not necessrily be integers in (0, ..., n_states - 1), for example.
        The semantics of ``mapping_[i] = j`` is that state ``i`` from the
        "input space" is represented by the index ``j`` in this MSM.
    countsmat_ : array_like, shape(n_states, n_states)
        Symmetrized transition counts. countsmat_[i, j] is the expected
        number of transitions from state i to state j after correcting
        for reversibly. The indices `i` and `j` are the "internal" indices
        described above.
    transmat_ : array_like, shape(n_states, n_states)
        Maximum likelihood estimate of the reversible transition matrix.
        The indices `i` and `j` are the "internal" indices described above.
    populations_ : array, shape(n_states)
        The equilibrium population (stationary eigenvector) of transmat_
    """

    def __init__(self, lag_time=1, n_timescales=None,
                 reversible_type='mle', ergodic_trim=True, trim_weight=0,
                 prior_counts=0, verbose=True):
        self.reversible_type = reversible_type
        self.ergodic_trim = ergodic_trim
        self.trim_weight = trim_weight
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.prior_counts = prior_counts
        self.verbose = verbose

        # Keep track of whether to recalculate eigensystem
        self.is_dirty = True
        # Cached results
        self._eigenvectors = None
        self._eigenvalues = None

    def fit(self, sequences, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        sequences : list
            List of integer sequences, each of which is one-dimensional
        y : unused parameter

        Returns
        -------
        self
        """
        # step 1. count the number of transitions
        raw_counts, mapping = _transition_counts(sequences, self.lag_time)

        if self.ergodic_trim:
            # step 2. restrict the counts to the maximal strongly ergodic
            # subgraph
            self.countsmat_, mapping2 = _strongly_connected_subgraph(
                raw_counts, self.trim_weight, self.verbose)
            self.mapping_ = _dict_compose(mapping, mapping2)
        else:
            # no ergodic trimming.
            self.countsmat_ = raw_counts
            self.mapping_ = mapping

        method_map = {
            'mle': self._fit_mle,
            'transpose': self._fit_transpose,
            'none': self._fit_asymetric,
        }
        try:
            # step 3. estimate transition matrix
            method =  method_map[str(self.reversible_type).lower()]
            self.transmat_, self.populations_ = method(self.countsmat_)
        except KeyError:
            raise ValueError('reversible_type must be one of %s: %s' % (
                ', '.join(available_reversible_type), self.reversible_type))

        return self

    def _fit_mle(self, counts):
        transmat, populations = _transmat_mle_prinz(
            counts + self.prior_counts)

        populations /= populations.sum(dtype=float)
        return transmat, populations

    def _fit_transpose(self, counts):
        rev_counts = 0.5 * (counts + counts.T) + self.prior_counts

        populations = rev_counts.sum(axis=0)
        populations /= populations.sum(dtype=float)
        transmat = rev_counts.astype(float) / rev_counts.sum(axis=1)[:, None]
        return transmat, populations

    def _fit_asymetric(self, counts):
        rc = counts + self.prior_counts
        transmat = rc.astype(float) / rc.sum(axis=1)[:, None]

        u, v = _eigs(transmat.T, k=1, which='LR')
        assert len(u) == 1

        populations = v[:, 0]
        populations /= populations.sum(dtype=float)

        return transmat, populations

    def score_ll(self, sequences):
        """log of the likelihood of sequences with respect to the model

        Parameters
        ----------
        sequences : list
            List of integer sequences, each of which is one-dimensional

        Returns
        -------
        loglikelihood : float
            The natural log of the likelihood, computed as
            :math:`\sum_{ij} C_{ij} \log(P_{ij})`
            where C is a matrix of counts computed from the input sequences.
        """
        counts, mapping = _transition_counts(sequences)
        if not set(self.mapping_.keys()).issuperset(mapping.keys()):
            return -np.inf
        inverse_mapping = {v:k for k, v in mapping.items()}

        # maps indices in counts to indices in transmat
        m2 = _dict_compose(inverse_mapping, self.mapping_)
        indices = [e[1] for e in sorted(m2.items())]

        transmat_slice = self.transmat_[np.ix_(indices, indices)]
        return np.nansum(np.log(transmat_slice) * counts)

    def _get_eigensystem(self):
        if not self.is_dirty:
            return self._eigenvalues, self._eigenvectors

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.transmat_.shape[0] - 3

        u, v = _eigs(self.transmat_.T, k=n_timescales + 1)

        order = np.argsort(-np.real(u))
        u = np.real_if_close(u[order])
        v = np.real_if_close(v[:, order])

        self._eigenvalues = u
        self._eigenvectors = v
        self.is_dirty = False

        return u, v

    @property
    def timescales_(self):
        u, v = self._get_eigensystem()

        # make sure to leave off equilibrium distribution
        timescales = - self.lag_time / np.log(u[1:])
        return timescales

    @property
    def eigenvectors_(self):
        u, v = self._get_eigensystem()
        return v

    @property
    def eigenvalues_(self):
        u, v = self._get_eigensystem()
        return u


def ndgrid_msm_likelihood_score(estimator, sequences):
    """Log-likelihood score function for an (NDGrid, MarkovStateModel) pipeline

    Parameters
    ----------
    estimator : sklearn.pipeline.Pipeline
        A pipeline estimator containing an NDGrid followed by a MarkovStateModel
    sequences: list of array-like, each of shape (n_samples_i, n_features)
        Data sequences, where n_samples_i in the number of samples
        in sequence i and n_features is the number of features.

    Returns
    -------
    log_likelihood : float
        Mean log-likelihood per data point.

    Examples
    --------
    >>> pipeline = Pipeline([
    >>>    ('grid', NDGrid()),
    >>>    ('msm', MarkovStateModel())
    >>> ])
    >>> grid = GridSearchCV(pipeline, param_grid={
    >>>    'grid__n_bins_per_feature': [10, 20, 30, 40]
    >>> }, scoring=ndgrid_msm_likelihood_score)
    >>> grid.fit(dataset)
    >>> print grid.grid_scores_

    References
    ----------
    .. [1] McGibbon, R. T., C. R. Schwantes, and V. S. Pande. "Statistical
       Model Selection for Markov Models of Biomolecular Dynamics." J. Phys.
       Chem B. (2014)
    """
    import msmbuilder.MSMLib as msmlib
    from mixtape import cluster
    grid = [model for (name, model) in estimator.steps if isinstance(model, cluster.NDGrid)][0]
    msm = [model for (name, model) in estimator.steps if isinstance(model, MarkovStateModel)][0]

    # NDGrid supports min/max being different along different directions, which
    # means that the bin widths are coordinate dependent. But I haven't
    # implemented that because I've only been using this for 1D data
    if grid.n_features != 1:
        raise NotImplementedError("file an issue on github :)")

    raise NotImplementedError()

    # transition_log_likelihood = 0
    # emission_log_likelihood = 0
    # logtransmat = np.nan_to_num(np.log(np.asarray(msm.transmat_.todense())))
    # width = grid.grid[0,1] - grid.grid[0,0]
    #
    # for X in grid.transform(sequences):
    #     counts = np.asarray(_apply_mapping_to_matrix(
    #         msmlib.get_counts_from_traj(X, n_states=grid.n_bins),
    #         msm.mapping_).todense())
    #     transition_log_likelihood += np.multiply(counts, logtransmat).sum()
    #     emission_log_likelihood += -1 * np.log(width) * len(X)
    #
    # return (transition_log_likelihood + emission_log_likelihood) / sum(len(x) for x in sequences)
    #

def _strongly_connected_subgraph(counts, weight=0, verbose=True):
    """Trim a transition count matrix down to its maximal
    strongly ergodic subgraph.

    From the counts matrix, we define a graph where there exists
    a directed edge between two nodes, `i` and `j` if
    `counts[i][j] > weight`. We then find the nodes belonging to the largest
    strongly connected subgraph of this graph, and return a new counts
    matrix formed by these rows and columns of the input `counts` matrix.

    Parameters
    ----------
    counts : np.array, shape=(n_states_in, n_states_in)
        Input set of directed counts.
    weight : float
        The cutoff criterion.
    verbose : bool
        Print a short statement

    Returns
    -------
    counts_component :
        "Trimmed" version of ``counts``, including only states in the
        maximal strongly ergodic subgraph.
    mapping : dict
        Mapping from "input" states indices to "output" state indices
        The semantics of ``mapping[i] = j`` is that state ``i`` from the
        "input space" for the counts matrix is represented by the index
        ``j`` in counts_component
    """
    n_states_input = counts.shape[0]
    n_components, component_assignments = scipy.sparse.csgraph.connected_components(
        scipy.sparse.csr_matrix(counts) > weight, connection="strong")
    populations = np.array(counts.sum(0)).flatten()
    component_pops = np.array([populations[component_assignments == i].sum() for i in range(n_components)])
    which_component = component_pops.argmax()

    if verbose:
        print("MSM contains %d strongly connected components "
              "above weight=%.2f. Component %d selected, with "
              "population %f%%" % (n_components, weight, which_component,
                                   100 * component_pops[which_component] / component_pops.sum()))

    # keys are all of the "input states" which have a valid mapping to the output.
    keys = np.arange(n_states_input)[component_assignments == which_component]
    # values are the "output" state that these guys are mapped to
    values = np.arange(len(keys))
    mapping = dict(zip(keys, values))
    n_states_output = len(mapping)

    trimmed_counts = np.zeros((n_states_output, n_states_output), dtype=counts.dtype)
    trimmed_counts[np.ix_(values, values)] = counts[np.ix_(keys, keys)]
    return trimmed_counts, mapping


def _transition_counts(sequences, lag_time=1):
    """Count the number of directed transitions in a collection of sequences
    in a discrete space.

    Parameters
    ----------
    sequences : list of array-like, each 1-dimensional
        Each element of sequences should be a separate timeseries of "labels",
        which can be integers, strings, etc.
    lag_time : int
        The time (index) delay for the counts.

    Returns
    -------
    counts : array, shape=(n_states, n_states)
        counts[i][j] counts the number of times a sequences was in state `i` at time
        t, and state `j` at time `t+self.lag_time`, over the full set of trajectories.
    mapping : dict
        Mapping from the items in the sequences to the indices in (0, n_states-1)
        used for the count matrix.

    Examples
    --------
    >>> sequence = [0, 0, 0, 1, 1]
    >>> counts, mapping = _transition_counts([sequence])
    >>> print counts
    [[2, 1],
     [0, 1]]
    >>> print mapping
    {0: 0, 1: 1}

    >>> sequence = [100, 200, 300]
    >>> counts, mapping = _transition_counts([sequence])
    >>> print counts
    [[ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 0.  0.  0.]]
    >>> print mapping
    {100: 0, 200: 1, 300: 2}
    """

    typed_sequences = []
    for y in sequences:
        if not hasattr(y, '__iter__'):
            raise ValueError('sequences must be a list of arrays')
        typed_sequences.append(column_or_1d(y, warn=True))

    classes = np.unique(np.concatenate(typed_sequences))
    n_states = len(classes)

    mapping = dict(zip(classes, np.arange(n_states)))
    mapping_is_identity = np.all(classes == np.arange(n_states))
    mapping_fn = np.vectorize(mapping.get)

    counts = np.zeros((n_states, n_states), dtype=float)
    for y in typed_sequences:
        from_states = y[: -lag_time: 1]
        to_states = y[lag_time::1]
        if not mapping_is_identity and len(from_states) > 0 and len(to_states) > 0:
            from_states = mapping_fn(from_states)
            to_states = mapping_fn(to_states)

        transitions = np.row_stack((from_states, to_states))
        C = scipy.sparse.coo_matrix(
            (np.ones(transitions.shape[1], dtype=int), transitions),
            shape=(n_states, n_states))
        counts = counts + np.asarray(C.todense())

    return counts, mapping


def _dict_compose(dict1, dict2):
    """
    Example
    -------
    >>> dict1 = {'a': 0, 'b': 1, 'c': 2}
    >>> dict2 = {0: 'A', 1: 'B'}
    >>> _dict_compose(dict1, dict2)
    {'a': 'A', 'b': 'b'}
    """
    return {k: dict2.get(v) for k, v in dict1.items() if v in dict2}


def _eigs(A, k=6, **kwargs):
    if 1 <= k <= A.shape[0] - 1:
        return scipy.sparse.linalg.eigs(A, k=k, **kwargs)
    return scipy.linalg.eig(A)

