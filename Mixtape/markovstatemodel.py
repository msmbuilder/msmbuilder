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

import warnings
import numpy as np
import scipy.sparse
import scipy.linalg
from mixtape.utils import list_of_1d
from sklearn.utils import column_or_1d, check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from mixtape._markovstatemodel import _transmat_mle_prinz

__all__ = ['MarkovStateModel']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class MarkovStateModel(BaseEstimator, TransformerMixin):
    """Reversible Markov State Model

    Parameters
    ----------
    lag_time : int
        The lag time of the model
    n_timescales : int, optional
        The number of dynamical timescales to calculate when diagonalizing
        the transition matrix.
    reversible_type : {'mle', 'transpose', None}
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization (BFGS), and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    ergodic_trim : bool, default=True
        Build a model using only the maximal strongly ergodic subgraph of the
        input data.
    trim_weight : int, default=1
        Threshold by which ergodicity is judged in the input data. Greater or
        equal to this many transition counts in both directions are required
        to include an edge in the ergodic subgraph.
    prior_counts : float, optional
        Add a number of "pseudo counts" to each entry in the counts matrix.
        When prior_counts == 0 (default), the assigned transition
        probability between two states with no observed transitions will be zero,
        whereas when prior_counts > 0, even this unobserved transitions will be
        given nonzero probability. Note that prior_counts _totally_ destroys
        performance when the number of states is large, because none of the
        matrices are sparse anymore.
    verbose : bool
        Enable verbose printout

    Attributes
    ----------
    n_states_ : int
        The number of states in the model
    mapping_ : dict
        Mapping between "input" labels and internal state indices used by the
        counts and transition matrix for this Markov state model. Input states
        need not necessrily be integers in (0, ..., n_states_ - 1), for example.
        The semantics of ``mapping_[i] = j`` is that state ``i`` from the
        "input space" is represented by the index ``j`` in this MSM.
    countsmat_ : array_like, shape = (n_states_, n_states_)
        Symmetrized transition counts. countsmat_[i, j] is the expected
        number of transitions from state i to state j after correcting
        for reversibly. The indices `i` and `j` are the "internal" indices
        described above.
    transmat_ : array_like, shape = (n_states_, n_states_)
        Maximum likelihood estimate of the reversible transition matrix.
        The indices `i` and `j` are the "internal" indices described above.
    populations_ : array, shape = (n_states_,)
        The equilibrium population (stationary eigenvector) of transmat_
    """

    def __init__(self, lag_time=1, n_timescales=10,
                 reversible_type='mle', ergodic_trim=True, trim_weight=1,
                 prior_counts=0, verbose=True):
        self.reversible_type = reversible_type
        self.ergodic_trim = ergodic_trim
        self.trim_weight = trim_weight
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.prior_counts = prior_counts
        self.verbose = verbose

        # Keep track of whether to recalculate eigensystem
        self._is_dirty = True
        # Cached results
        self._eigenvectors = None
        self._eigenvalues = None

    def fit(self, sequences, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        Returns
        -------
        self

        Notes
        -----
        `None` and `NaN` are recognized immediately as invalid labels.
        Therefore, transition counts from or to a sequence item which is NaN or
        None will not be counted. The mapping_ attribute will not include the
        NaN or None.
        """
        sequences = list_of_1d(sequences)
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

        self.n_states_ = self.countsmat_.shape[0]

        # use a dict like a switch statement: dispatch to different
        # transition matrix estimators depending on the value of
        # self.reversible_type
        fit_method_map = {
            'mle': self._fit_mle,
            'transpose': self._fit_transpose,
            'none': self._fit_asymetric,
        }
        try:
            # pull out the appropriate method
            fit_method =  method_map[str(self.reversible_type).lower()]
            # step 3. estimate transition matrix
            self.transmat_, self.populations_ = fit_method(self.countsmat_)
        except KeyError:
            raise ValueError('reversible_type must be one of %s: %s' % (
                ', '.join(fit_method_map.keys()), self.reversible_type))

        self._is_dirty = True
        return self

    def _fit_mle(self, counts):
        if not self.ergodic_trim:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                warnings.warn("reversible_type='mle' and ergodic_trim=False are incompatibile")

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

        u, lv, rv = _eigs(transmat, k=1, which='LR')
        assert len(u) == 1

        populations = lv[:, 0]
        populations /= populations.sum(dtype=float)

        return transmat, populations

    def transform(self, sequences, mode='clip'):
        """Transform a list of sequences to internal indexing

        Recall that `sequences` can be arbitrary labels, whereas `transmat_` and
        `countsmat_` are indexed with integers between 0 and `n_states` - 1.
        This methods maps a set of sequences from the labels into this internal
        indexing.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.
        mode : {'clip', 'fill'}
            Method by which to treat labels in `sequences` which do not have
            a corresponding index. This can be due, for example, to the ergodic
            trimming step.

           ``clip``
               Unmapped labels are removed during transform. If they occur
               at the beginning or end of a sequence, the resulting transformed
               sequence will be shorted. If they occur in the middle of a
               sequence, that sequence will be broken into two (or more)
               sequences. (Default)
            ``fill``
               Unmapped labels will be replaced with NaN, to signal missing
               data. [The use of NaN to signal missing data is not fantastic,
               but it's consistent with current behavior of the ``pandas``
               library.]

        Returns
        -------
        mapped_sequences : list
            List of sequences in internal indexing
        """
        if not mode in ['clip', 'fill']:
            raise ValueError('mode must be one of ["clip", "fill"]: %s' % mode)
        sequences = list_of_1d(sequences)

        f = np.vectorize(lambda k: self.mapping_.get(k, np.nan), otypes=[np.float])

        result = []
        for y in sequences:
            a = f(y)
            if mode == 'fill':
                if np.all(np.mod(a, 1) ==  0):
                    result.append(a.astype(int))
                else:
                    result.append(a)
            elif mode == 'clip':
                result.extend([a[s].astype(int) for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))])
            else:
                raise RuntimeError()

        return result

    def inverse_transform(self, sequences):
        """Transform a list of sequences from internal indexing into
        labels

        Parameters
        ----------
        sequences : list
            List of sequences, each of which is one-dimensional array of integers
            in 0, ..., n_states_ - 1.

        Returns
        -------
        sequences : list
            List of sequences, each of which is one-dimensional array of labels.
        """
        sequences = list_of_1d(sequences)
        inverse_mapping = {v: k for k, v in self.mapping_.items()}
        f = np.vectorize(inverse_mapping.get)

        result = []
        for y in sequences:
            uq = np.unique(y)
            if not np.all(np.logical_and(0 <= uq, uq < self.n_states_)):
                raise ValueError('sequence must be between 0 and n_states-1')

            result.append(f(y))
        return result

    def eigtransform(self, sequences, right=True, mode='clip'):
        """Transform a list of sequences by projecting the sequences onto
        the first `n_timescales` dynamical eigenvectors.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        right : bool
            Which eigenvectors to map onto. Both the left (:math:`\Phi`) and
            the right (:math`\Psi`) eigenvectors of the transition matrix are
            commonly used, and differ in their normalization. The two sets of
            eigenvectors are related by the stationary distribution ::

                \Phi_i(x) = \Psi_i(x) * \mu(x)

        mode : {'clip', 'fill'}
            Method by which to treat labels in `sequences` which do not have
            a corresponding index. This can be due, for example, to the ergodic
            trimming step.

           ``clip``
               Unmapped labels are removed during transform. If they occur
               at the beginning or end of a sequence, the resulting transformed
               sequence will be shorted. If they occur in the middle of a
               sequence, that sequence will be broken into two (or more)
               sequences. (Default)
            ``fill``
               Unmapped labels will be replaced with NaN, to signal missing
               data. [The use of NaN to signal missing data is not fantastic,
               but it's consistent with current behavior of the ``pandas``
               library.]

        Returns
        -------
        transformed : list of 2d arrays
            Each element of transformed is an array of shape ``(n_samples,
            n_timescales)`` containing the transformed data.

        References
        ----------
        .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J. Chem. Phys. 134.17 (2011): 174105.
        """

        result = []
        for y in self.transform(sequences, mode=mode):
            if right:
                op = self.right_eigenvectors_[:, 1:]
            else:
                op = self.left_eigenvectors_[:, 1:]
            result.append(np.take(op, y))

        return result

    def sample(self, state=None, n_steps=100, random_state=None):
        """Generate a random sequence of states by propagating the model

        Parameters
        ----------
        state : {None, ndarray, label}
            Specify the starting state for the chain.

            ``None``
                Choose the initial state by randomly drawing from the model's
                stationary distribution.
            ``array-like``
                If ``state`` is a 1D array with length equal to ``n_states_``,
                then it is is interpreted as an initial multinomial distribution
                from which to draw the chain's initial state. Note that the indexing
                semantics of this array must match the _internal_ indexing of
                this model.
            otherwise
                Otherwise, ``state`` is interpreted as a particular
                deterministic state label from which to begin the trajectory.
        n_steps : int
            Lengths of the resulting trajectory
        random_state : int or RandomState instance or None (default)
            Pseudo Random Number generator seed control. If None, use the
            numpy.random singleton.

        Returns
        -------
        sequence : array of length n_steps
            A randomly sampled label sequence
        """
        random = check_random_state(random_state)
        r = random.rand(1 + n_steps)

        if state is None:
            initial = np.sum(np.cumsum(self.populations_) < r[0])
        elif hasattr(state, '__len__') and len(state) == self.n_states:
            initial = np.sum(np.cumsum(state) < r[0])
        else:
            initial = self.mapping_[state]

        cstr = np.cumsum(self.transmat_, axis=1)

        chain = [initial]
        for i in range(1, n_steps):
            chain.append(np.sum(cstr[chain[i-1], :] < r[i]))

        return self.inverse_transform([chain])[0]

    def score_ll(self, sequences):
        """log of the likelihood of sequences with respect to the model

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

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
        if not self._is_dirty:
            return self._eigenvalues, self._left_eigenvectors, self._right_eigenvectors

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_states_ - 1

        u, lv, rv = _eigs(self.transmat_, k=n_timescales + 1, left=True, right=True)

        order = np.argsort(-np.real(u))
        u = np.real_if_close(u[order])
        lv = np.real_if_close(lv[:, order])
        rv = np.real_if_close(rv[:, order])

        # TODO: Normalize lv and rv correctly.

        self._eigenvalues = u
        self._left_eigenvectors = lv
        self._right_eigenvectors = rv

        self._is_dirty = False

        return u, lv, rv

    @property
    def timescales_(self):
        """Implied relaxation timescales of the model.

        [TODO]
        """
        u, lv, rv = self._get_eigensystem()

        # make sure to leave off equilibrium distribution
        timescales = - self.lag_time / np.log(u[1:])
        return timescales

    @property
    def eigenvalues_(self):
        """Eigenvalues of the transition matrix.
        """
        u, lv, rv = self._get_eigensystem()
        return u

    @property
    def left_eigenvectors_(self):
        """Left eigenvectors, :math:`\Phi`, of the transition matrix.

        TODO: describe normalization
        """
        u, lv, rv = self._get_eigensystem()
        return lv

    @property
    def right_eigenvectors_(self):
        """Right eigenvectors, :math:`\Psi`, of the transition matrix.

        TODO: describe normalization
        """
        u, lv, rv = self._get_eigensystem()
        return rv

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

def _strongly_connected_subgraph(counts, weight=1, verbose=True):
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
        Threshold by which ergodicity is judged in the input data. Greater or
        equal to this many transition counts in both directions are required
        to include an edge in the ergodic subgraph.
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
        scipy.sparse.csr_matrix(counts >= weight), connection="strong")
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

    if n_components == n_states_input and counts[np.ix_(keys, keys)] == 0:
        # if we have a completely disconnected graph with no self-transitions
        return np.zeros((0,0)), {}

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
    sequences : list of array-like
        List of sequences, or a single sequence. Each sequence should be a
        1D iterable of state labels. Labels can be integers, strings, or
        other orderable objects.
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

    Notes
    -----
    `None` and `NaN` are recognized immediately as invalid labels. Therefore,
    transition counts from or to a sequence item which is NaN or None will not
    be counted. The mapping return value will not include the NaN or None.
    """
    classes = np.unique(np.concatenate(sequences))
    contains_nan = (classes.dtype.kind == 'f') and np.any(np.isnan(classes))
    contains_none = any(c is None for c in classes)

    if contains_nan:
        classes = classes[~np.isnan(classes)]
    if contains_none:
        classes = [c for c in classes if c is not None]

    n_states = len(classes)

    mapping = dict(zip(classes, range(n_states)))
    mapping_is_identity = np.all(classes == np.arange(n_states))
    mapping_fn = np.vectorize(mapping.get, otypes=[np.int])
    none_to_nan = np.vectorize(lambda x: np.nan if x is None else x, otypes=[np.float])

    counts = np.zeros((n_states, n_states), dtype=float)
    for y in sequences:
        from_states = y[: -lag_time: 1]
        to_states = y[lag_time::1]

        if contains_none:
            from_states = none_to_nan(from_states)
            to_states = none_to_nan(to_states)

        if contains_nan or contains_none:
            # mask out nan in either from_states or to_states
            mask = ~(np.isnan(from_states) + np.isnan(to_states))
            from_states = from_states[mask]
            to_states = to_states[mask]

        if (not mapping_is_identity) and len(from_states) > 0 and len(to_states) > 0:
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
    if 1 <= k < A.shape[0] - 1:
        u, rv = scipy.sparse.linalg.eigs(A, k=k, **kwargs)
        u, lv = scipy.sparse.linalg.eigs(A.T, k=k, **kwargs)

    u, lv, rv = scipy.linalg.eig(A, left=True, right=True)
    indices = np.argsort(-np.real(u))

    u = u[indices[:k]]
    lv = lv[:, indices[:k]]
    rv = rv[:, indices[:k]]

    return u, lv, rv
