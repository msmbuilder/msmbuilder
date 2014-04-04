from __future__ import print_function, division, absolute_import

import time
import warnings
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator
from mdtraj.utils import ensure_type
from mixtape import _reversibility

__all__ = ['MarkovStateModel']


class MarkovStateModel(BaseEstimator):

    """Reversible Markov State Model

    Parameters
    ----------
    n_states : int
        The number of states in the model
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

    Attributes
    ----------
    mapping_ : dict
        Mapping between "input" states and internal state indices for this
        Markov state model. The indexing of the states in the labeled
        sequences supplied to fit() may have gaps (i.e. the unique indices
        are not continuous integers starting from zero), or, if
        `ergodic_trim==True`, some of those "input states" may be excluded
        from the model because they do not lie in the maximal ergodic subgraph.
        In either case, the semantics of `mapping_[i] = j` is that state `i`
        from the "input space" is represented by the index `j` in this Markov
        state model.
    rawcounts_ : array_like, shape(n_states, n_states)
        Unsymmetrized transition counts. rawcounts_[i, j] is the observed
        number of transitions from state i to state j. The indices `i` and
        `j` are the "internal" indices described above.
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

    def __init__(self, n_states, reversible_type='mle', ergodic_trim=True):
        self.n_states = n_states
        self.reversible_type = reversible_type
        self.ergodic_trim = ergodic_trim

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
        from msmbuilder.msm_analysis import get_eigenvectors
        from msmbuilder.MSMLib import mle_reversible_count_matrix, estimate_transition_matrix, ergodic_trim

        self.rawcounts_ = self._count_transitions(sequences)

        # STEP (1): Ergodic trimming
        if self.ergodic_trim:
            self.rawcounts_, mapping = ergodic_trim(scipy.sparse.csr_matrix(self.rawcounts_))
            self.mapping_ = {}
            for i, j in enumerate(mapping):
                if j != -1:
                    self.mapping_[i] = j
        else:
            self.mapping_ = dict(zip(np.arange(self.n_states), np.arange(self.n_states)))

        # STEP (2): Reversible counts matrix
        if self.reversible_type in ['mle', 'MLE']:
            self.countsmat_ = mle_reversible_count_matrix(self.rawcounts_)
        elif self.reversible_type in ['transpose', 'Transpose']:
            self.countsmat_ = 0.5 * (self.rawcounts_ + self.rawcounts_.T)
        elif self.reversible_type is None:
            self.countsmat_ = self.rawcounts_
        else:
            raise RuntimeError()

        # STEP (3): transition matrix
        self.transmat_ = estimate_transition_matrix(self.countsmat_)

        # STEP (3.5): Stationary eigenvector
        if self.reversible_type in ['mle', 'MLE', 'transpose', 'Transpose']:
            self.populations_ = np.array(self.countsmat_.sum(0)).flatten()
        elif self.reversible_type is None:
            vectors = get_eigenvectors(self.transmat_, 5)[1]
            self.populations_ = vectors[:, 0]
        else:
            raise RuntimeError()
        self.populations_ /= self.populations_.sum()  # ensure normalization

        return self

    def _count_transitions(self, sequences):
        counts = scipy.sparse.coo_matrix((self.n_states, self.n_states), dtype=np.float32)

        for sequence in sequences:
            from_states = sequence[: -1: 1]
            to_states = sequence[1::1]
            transitions = np.row_stack((from_states, to_states))
            C = scipy.sparse.coo_matrix((np.ones(transitions.shape[1], dtype=int), transitions), shape=(self.n_states, self.n_states))
            counts = counts + C

        return counts

    def timescales_(self, n_timescales=None):
        from msmbuilder.msm_analysis import get_reversible_eigenvectors, get_eigenvectors
        if n_timescales is None:
            n_timescales = self.n_states - 1

        n_eigenvectors = n_timescales + 1
        if self.reversible_type in ['mle', 'MLE', 'transpose', 'Transpose'] and self.transmat_.shape[0] > 50:
            e_values = get_reversible_eigenvectors(self.transmat_, n_eigenvectors, populations=self.populations_)[0]
        else:
            e_values = get_eigenvectors(self.transmat_, n_eigenvectors, epsilon=1)[0]

        # make sure to leave off equilibrium distribution
        timescales = -1 / np.log(e_values[1 : n_eigenvectors])
        return timescales
