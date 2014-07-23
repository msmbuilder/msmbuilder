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

    Attributes
    ----------
    transmat_ : array_like, shape(n_states, n_states)
        Maximum likelihood estimate of the reversible transition matrix.
        The indices `i` and `j` are the "internal" indices described above.
    populations_ : array, shape(n_states)
        The equilibrium population (stationary eigenvector) of transmat_
    mapping_ : dict
        Mapping between "input" states and internal state indices for this
        Markov state model.  This is necessary because of ergodic_trim.
        The semantics of ``mapping_[i] = j`` is that state ``i`` from the
        "input space" is represented by the index ``j`` in this MSM.
    rawcounts_ : array_like, shape(n_states, n_states)
        Unsymmetrized transition counts. rawcounts_[i, j] is the observed
        number of transitions from state i to state j. The indices `i` and
        `j` are the "internal" indices described above.
    countsmat_ : array_like, shape(n_states, n_states)
        Symmetrized transition counts. countsmat_[i, j] is the expected
        number of transitions from state i to state j after correcting
        for reversibly. The indices `i` and `j` are the "internal" indices
        described above.

    """

    def __init__(self, n_states=None, lag_time=1, n_timescales=None,
                 reversible_type='mle', ergodic_trim=True, trim_weight=0,
                 prior_counts=0):
        self.reversible_type = reversible_type
        self.ergodic_trim = ergodic_trim
        self.trim_weights = trim_weight
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.prior_counts = prior_counts

        # Keep track of whether to recalculate eigensystem
        self.is_dirty = True
        # Cached results
        self._eigenvectors = None
        self._eigenvalues = None

        available_reversible_type = ['mle', 'MLE', 'transpose', 'Transpose', None]
        if self.reversible_type not in available_reversible_type:
            raise ValueError('symmetrize must be one of %s: %s' % (
                ', '.join(available_reversible_type), reversible_type))

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
        counts = self._count_transitions(sequences)
        if self.ergodic_trim:
            self.counts_, self.mapping_ = _strongly_connected_subgraph(counts, self.trim_weight)
        else:
            self.counts_ = counts
            self.mapping_ = dict(zip(np.unique(np.concatenate(sequences)),
                                     range(self.counts_.shape[0])))

        if self.reversible_type in ['mle', 'MLE']:
            self.transmat_, self.populations_ = _transmat_mle_prinz(trimmed_counts + self.prior_counts)
        elif self.reversible_type in ['transpose', 'Transpose']:
            rc = 0.5 * (trimmed_counts + trimmed_counts.T) + self.prior_counts
            self.populations_ = rc.sum(axis=0)
            self.transmat_ = rc.astype(float) / rc.sum(axis=1)[:, None]
        elif self.reversible_type is None:
            rc = trimmed_counts + self.prior_counts
            self.transmat_ = rc.astype(float) / rc.sum(axis=1)[:, None]
        else:
            raise RuntimeError()

        if self.reversible_type is None:
            vectors = get_eigenvectors(self.transmat_, 5)[1]
            self.populations_ = vectors[:, 0]

        self.populations_ /= self.populations_.sum(dtype=float)
        self.is_dirty = True

        return self

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
        counts = self._count_transitions(sequences)

        if not scipy.sparse.isspmatrix(self.transmat_):
            transition_matrix = scipy.sparse.csr_matrix(self.transmat_)
        else:
            transition_matrix = self.transmat_.tocsr()
        row, col = counts.nonzero()

        return np.sum(np.log(np.asarray(transition_matrix[row, col]))
                      * np.asarray(counts[row, col]))


    def _get_eigensystem(self):
        if not self.is_dirty:
            return self._eigenvalues, self._eigenvectors

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.transmat_.shape[0] - 3

        u, v = scipy.sparse.linalg.eigs(self.transmat_.transpose(),
                                        k=n_timescales + 1)
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

    transition_log_likelihood = 0
    emission_log_likelihood = 0
    logtransmat = np.nan_to_num(np.log(np.asarray(msm.transmat_.todense())))
    width = grid.grid[0,1] - grid.grid[0,0]

    for X in grid.transform(sequences):
        counts = np.asarray(_apply_mapping_to_matrix(
            msmlib.get_counts_from_traj(X, n_states=grid.n_bins),
            msm.mapping_).todense())
        transition_log_likelihood += np.multiply(counts, logtransmat).sum()
        emission_log_likelihood += -1 * np.log(width) * len(X)

    return (transition_log_likelihood + emission_log_likelihood) / sum(len(x) for x in sequences)


def _apply_mapping_to_matrix(mat, mapping):
    ndim_new = np.max(mapping.values()) + 1
    mat_new = scipy.sparse.dok_matrix((ndim_new, ndim_new))
    for (i, j), e in mat.todok().items():
        try:
            mat_new[mapping[i], mapping[j]] = e
        except KeyError:
            pass
    return mat_new

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
    in a discrete space

    Parameters
    ----------
    sequences : list of array-like, each 1-dimensional

    Returns
    -------
    counts : array, shape=(n_states, n_states)
        counts[i][j] counts the number of times a sequences was in state `i` at time
        t, and state `j` at time `t+self.lag_time`, over the full set of trajectories.
    mapping :
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
        if not mapping_is_identity:
            from_states = mapping_fn(from_states)
            to_states = mapping_fn(to_states)

        transitions = np.row_stack((from_states, to_states))
        C = scipy.sparse.coo_matrix(
            (np.ones(transitions.shape[1], dtype=int), transitions),
            shape=(n_states, n_states))
        counts = counts + C.todense()

    return counts, mapping
