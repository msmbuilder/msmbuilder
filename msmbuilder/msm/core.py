# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import collections

import numpy as np
import scipy.linalg
from scipy.sparse import csgraph, csr_matrix, coo_matrix
from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state

from . import _ratematrix
from ..utils import list_of_1d

__all__ = [
    '_MappingTransformMixin', '_dict_compose', '_strongly_connected_subgraph',
    '_transition_counts', '_solve_ratemat_eigensystem',
    '_normalize_eigensystem',
    '_solve_msm_eigensystem',
]


class _MappingTransformMixin(TransformerMixin):

    def partial_transform(self, sequence, mode='clip'):
        """Transform a sequence to internal indexing

        Recall that `sequence` can be arbitrary labels, whereas ``transmat_``
        and ``countsmat_`` are indexed with integers between 0 and
        ``n_states - 1``. This methods maps a set of sequences from the labels
        onto this internal indexing.

        Parameters
        ----------
        sequence : array-like
            A 1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.
        mode : {'clip', 'fill'}
            Method by which to treat labels in `sequence` which do not have
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
        mapped_sequence : list or ndarray
            If mode is "fill", return an ndarray in internal indexing.
            If mode is "clip", return a list of ndarrays each in internal
            indexing.
        """
        if mode not in ['clip', 'fill']:
            raise ValueError('mode must be one of ["clip", "fill"]: %s' % mode)
        sequence = np.asarray(sequence)
        if sequence.ndim != 1:
            raise ValueError("Each sequence must be 1D")

        f = np.vectorize(lambda k: self.mapping_.get(k, np.nan),
                         otypes=[np.float])

        a = f(sequence)
        if mode == 'fill':
            if np.all(np.mod(a, 1) == 0):
                result = a.astype(int)
            else:
                result = a
        elif mode == 'clip':
            result = [a[s].astype(int) for s in
                      np.ma.clump_unmasked(np.ma.masked_invalid(a))]
        else:
            raise RuntimeError()

        return result

    def transform(self, sequences, mode='clip'):
        """Transform a list of sequences to internal indexing

        Recall that `sequences` can be arbitrary labels, whereas ``transmat_``
        and ``countsmat_`` are indexed with integers between 0 and
        ``n_states - 1``. This methods maps a set of sequences from the labels
        onto this internal indexing.

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
        if mode not in ['clip', 'fill']:
            raise ValueError('mode must be one of ["clip", "fill"]: %s' % mode)
        sequences = list_of_1d(sequences)

        result = []
        for y in sequences:
            if mode == 'fill':
                result.append(self.partial_transform(y, mode))
            elif mode == 'clip':
                result.extend(self.partial_transform(y, mode))
            else:
                raise RuntimeError()

        return result

    def _parse_ergodic_cutoff(self):
        """Get a numeric value from the ergodic_cutoff input,
        which can be 'on' or 'off'.
        """
        ec_is_str = isinstance(self.ergodic_cutoff, str)
        if ec_is_str and self.ergodic_cutoff.lower() == 'on':
            if self.sliding_window:
                return 1.0 / self.lag_time
            else:
                return 1.0
        elif ec_is_str and self.ergodic_cutoff.lower() == 'off':
            return 0.0
        else:
            return self.ergodic_cutoff

    def inverse_transform(self, sequences):
        """Transform a list of sequences from internal indexing into
        labels

        Parameters
        ----------
        sequences : list
            List of sequences, each of which is one-dimensional array of
            integers in ``0, ..., n_states_ - 1``.

        Returns
        -------
        sequences : list
            List of sequences, each of which is one-dimensional array
            of labels.
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

class _CountsMSMMixin(object):
    def _build_counts(self, sequences, y=None):
        sequences = list_of_1d(sequences)
        # step 1. count the number of transitions
        if int(self.lag_time) < 1:
            raise ValueError('Invalid lag_time: %s. \
            Lag_time must be >= 1' % self.lag_time)
        raw_counts, mapping = _transition_counts(
                sequences, int(self.lag_time),
                sliding_window=self.sliding_window)

        #step 2. Get the ergodic cutoff
        ergodic_cutoff = self._parse_ergodic_cutoff()
        # if sliding window, we are gonna have  a float cutoff
        if ergodic_cutoff > 0:
            # step 2. restrict the counts to the maximal strongly ergodic
            # subgraph
            self.countsmat_, mapping2, self.percent_retained_ = \
                _strongly_connected_subgraph(raw_counts, ergodic_cutoff,
                                             self.verbose)
            self.mapping_ = _dict_compose(mapping, mapping2)
        else:
            # no ergodic trimming.
            self.countsmat_ = raw_counts
            self.mapping_ = mapping
            self.percent_retained_ = 100
        self.n_states_ = self.countsmat_.shape[0]
        return

class _SampleMSMMixin(object):
    """Provides msm.sample() for drawing samples from continuous and discrete time MSMs."""

    def sample_discrete(self, state=None, n_steps=100, random_state=None):
        r"""Generate a random sequence of states by propagating the model
        using discrete time steps given by the model lagtime.

        Parameters
        ----------
        state : {None, ndarray, label}
            Specify the starting state for the chain.

            ``None``
                Choose the initial state by randomly drawing from the model's
                stationary distribution.
            ``array-like``
                If ``state`` is a 1D array with length equal to ``n_states_``,
                then it is is interpreted as an initial multinomial
                distribution from which to draw the chain's initial state.
                Note that the indexing semantics of this array must match the
                _internal_ indexing of this model.
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
        elif hasattr(state, '__len__') and len(state) == self.n_states_:
            initial = np.sum(np.cumsum(state) < r[0])
        else:
            initial = self.mapping_[state]

        cstr = np.cumsum(self.transmat_, axis=1)

        chain = [initial]
        for i in range(1, n_steps):
            chain.append(np.sum(cstr[chain[i - 1], :] < r[i]))

        return self.inverse_transform([chain])[0]

    def draw_samples(self, sequences, n_samples, random_state=None):
        """Sample conformations for a sequences of states.

        Parameters
        ----------
        sequences : list or list of lists
            A sequence or list of sequences, in which each element corresponds
            to a state label.
        n_samples : int
            How many samples to return for any given state.

        Returns
        -------
        selected_pairs_by_state : np.array, dtype=int,
            shape=(n_states, n_samples, 2) selected_pairs_by_state[state] gives
            an array of randomly selected (trj, frame) pairs from the specified
            state.

        See Also
        --------
        utils.map_drawn_samples : Extract conformations from MD trajectories by
        index.

        """
        if not any([isinstance(seq, collections.Iterable)
                    for seq in sequences]):
            sequences = [sequences]

        random = check_random_state(random_state)

        selected_pairs_by_state = []
        for state in range(self.n_states_):
            all_frames = [np.where(a == state)[0] for a in sequences]
            pairs = [(trj, frame) for (trj, frames) in enumerate(all_frames)
                     for frame in frames]
            if pairs:
                selected_pairs_by_state.append(
                        [pairs[random.choice(len(pairs))]
                         for i in range(n_samples)])
            else:
                selected_pairs_by_state.append([])

        return np.array(selected_pairs_by_state)


def _solve_ratemat_eigensystem(theta, k, n):
    """Find the dominant eigenpairs of a reversible rate matrix (master
    equation)

    Parameters
    ----------
    theta : ndarray, shape=(n_params,)
        The free parameters of the rate matrix
    k : int
        The number of eigenpairs to find
    n : int
        The number of states

    Notes
    -----
    Normalize the left (:math:`\phi`) and right (:math:``\psi``) eigenfunctions
    according to the following criteria.
      * The first left eigenvector, \phi_1, _is_ the stationary
        distribution, and thus should be normalized to sum to 1.
      * The left-right eigenpairs should be biorthonormal:
        <\phi_i, \psi_j> = \delta_{ij}
      * The left eigenvectors should satisfy
        <\phi_i, \phi_i>_{\mu^{-1}} = 1
      * The right eigenvectors should satisfy <\psi_i, \psi_i>_{\mu} = 1

    Returns
    -------
    eigvals : np.ndarray, shape=(k,)
        The largest `k` eigenvalues
    lv : np.ndarray, shape=(n_states, k)
        The normalized left eigenvectors (:math:`\phi`) of the rate matrix.
    rv :  np.ndarray, shape=(n_states, k)
        The normalized right eigenvectors (:math:`\psi`) of the rate matrix.
    """
    S = np.zeros((n, n))
    pi = np.exp(theta[-n:])
    pi = pi / pi.sum()

    _ratematrix.build_ratemat(theta, n, S, which='S')
    u, lv, rv = map(np.asarray, _ratematrix.eig_K(S, n, pi, 'S'))
    order = np.argsort(-u)
    u = u[order[:k]]
    lv = lv[:, order[:k]]
    rv = rv[:, order[:k]]

    return _normalize_eigensystem(u, lv, rv)


def _solve_msm_eigensystem(transmat, k):
    """Find the dominant eigenpairs of an MSM transition matrix

    Parameters
    ----------
    transmat : np.ndarray, shape=(n_states, n_states)
        The transition matrix
    k : int
        The number of eigenpairs to find.

    Notes
    -----
    Normalize the left (:math:`\phi`) and right (:math:``\psi``) eigenfunctions
    according to the following criteria.
      * The first left eigenvector, \phi_1, _is_ the stationary
        distribution, and thus should be normalized to sum to 1.
      * The left-right eigenpairs should be biorthonormal:
        <\phi_i, \psi_j> = \delta_{ij}
      * The left eigenvectors should satisfy
        <\phi_i, \phi_i>_{\mu^{-1}} = 1
      * The right eigenvectors should satisfy <\psi_i, \psi_i>_{\mu} = 1

    Returns
    -------
    eigvals : np.ndarray, shape=(k,)
        The largest `k` eigenvalues
    lv : np.ndarray, shape=(n_states, k)
        The normalized left eigenvectors (:math:`\phi`) of ``transmat``
    rv :  np.ndarray, shape=(n_states, k)
        The normalized right eigenvectors (:math:`\psi`) of ``transmat``
    """
    u, lv, rv = scipy.linalg.eig(transmat, left=True, right=True)
    order = np.argsort(-np.real(u))
    u = np.real_if_close(u[order[:k]])
    lv = np.real_if_close(lv[:, order[:k]])
    rv = np.real_if_close(rv[:, order[:k]])
    return _normalize_eigensystem(u, lv, rv)


def _normalize_eigensystem(u, lv, rv):
    """Normalize the eigenvectors of a reversible Markov state model according
    to our preferred scheme.
    """
    # first normalize the stationary distribution separately
    lv[:, 0] = lv[:, 0] / np.sum(lv[:, 0])

    for i in range(1, lv.shape[1]):
        # the remaining left eigenvectors to satisfy
        # <\phi_i, \phi_i>_{\mu^{-1}} = 1
        lv[:, i] = lv[:, i] / np.sqrt(np.dot(lv[:, i], lv[:, i] / lv[:, 0]))

    for i in range(rv.shape[1]):
        # the right eigenvectors to satisfy <\phi_i, \psi_j> = \delta_{ij}
        rv[:, i] = rv[:, i] / np.dot(lv[:, i], rv[:, i])

    return u, lv, rv


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
    n_components, component_assignments = csgraph.connected_components(
            csr_matrix(counts >= weight), connection="strong")
    populations = np.array(counts.sum(0)).flatten()
    component_pops = np.array([populations[component_assignments == i].sum() for
                               i in range(n_components)])
    which_component = component_pops.argmax()

    def cpop(which):
        csum = component_pops.sum()
        return 100 * component_pops[which] / csum if csum != 0 else np.nan

    percent_retained = cpop(which_component)
    if verbose:
        print("MSM contains %d strongly connected component%s "
              "above weight=%.2f. Component %d selected, with "
              "population %f%%" % (
              n_components, 's' if (n_components != 1) else '',
              weight, which_component, percent_retained))

    # keys are all of the "input states" which have a valid mapping to the output.
    keys = np.arange(n_states_input)[component_assignments == which_component]

    if n_components == n_states_input and counts[np.ix_(keys, keys)] == 0:
        # if we have a completely disconnected graph with no self-transitions
        return np.zeros((0, 0)), {}, percent_retained

    # values are the "output" state that these guys are mapped to
    values = np.arange(len(keys))
    mapping = dict(zip(keys, values))
    n_states_output = len(mapping)

    trimmed_counts = np.zeros((n_states_output, n_states_output),
                              dtype=counts.dtype)
    trimmed_counts[np.ix_(values, values)] = counts[np.ix_(keys, keys)]
    return trimmed_counts, mapping, percent_retained


def _transition_counts(sequences, lag_time=1, sliding_window=True):
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
    sliding_window : bool
        When lag_time > 1, consider *all*
        ``N = lag_time`` strided sequences starting from index
         0, 1, 2, ..., ``lag_time - 1``. The total, raw counts will
         be divided by ``N``. When this is False, only start from index 0.

    Returns
    -------
    counts : array, shape=(n_states, n_states)
        ``counts[i][j]`` counts the number of times a sequences was in state
        `i` at time t, and state `j` at time `t+self.lag_time`, over the
        full set of trajectories.
    mapping : dict
        Mapping from the items in the sequences to the indices in
        ``(0, n_states-1)`` used for the count matrix.

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
    if (not sliding_window) and lag_time > 1:
        return _transition_counts([X[::lag_time] for X in sequences],
                                  lag_time=1)

    classes = np.unique(np.concatenate(sequences))
    contains_nan = (classes.dtype.kind == 'f') and np.any(np.isnan(classes))
    contains_none = any(c is None for c in classes)

    if contains_nan:
        classes = classes[~np.isnan(classes)]
    if contains_none:
        classes = [c for c in classes if c is not None]

    n_states = len(classes)

    mapping = dict(zip(classes, range(n_states)))
    mapping_is_identity = (not contains_nan
                           and not contains_none
                           and classes.dtype.kind == 'i'
                           and np.all(classes == np.arange(n_states)))
    mapping_fn = np.vectorize(mapping.get, otypes=[np.int])
    none_to_nan = np.vectorize(lambda x: np.nan if x is None else x,
                               otypes=[np.float])

    counts = np.zeros((n_states, n_states), dtype=float)
    _transitions = []

    for y in sequences:
        y = np.asarray(y)
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

        if (not mapping_is_identity) and len(from_states) > 0 and len(
                to_states) > 0:
            from_states = mapping_fn(from_states)
            to_states = mapping_fn(to_states)

        _transitions.append(np.row_stack((from_states, to_states)))

    transitions = np.hstack(_transitions)
    C = coo_matrix((np.ones(transitions.shape[1], dtype=int), transitions),
                   shape=(n_states, n_states))
    counts = counts + np.asarray(C.todense())

    # If sliding window is False, this function will be called recursively
    # with strided trajectories and lag_time = 1, which gives the desired
    # number of counts. If sliding window is True, the counts are divided
    # by the "number of windows" (i.e. the lag_time). Count magnitudes
    # will be comparable between sliding-window and non-sliding-window cases.
    # If lag_time = 1, sliding_window makes no difference.
    counts /= float(lag_time)

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
