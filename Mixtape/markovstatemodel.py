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
from sklearn.base import BaseEstimator
from mdtraj.utils import ensure_type
from mixtape import _reversibility

__all__ = ['MarkovStateModel']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class MarkovStateModel(BaseEstimator):
    """Reversible Markov State Model

    Parameters
    ----------
    n_states : int
        The number of states in the model
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
                 reversible_type='mle', ergodic_trim=True, prior_counts=0):
        self.n_states = n_states
        self.reversible_type = reversible_type
        self.ergodic_trim = ergodic_trim
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.prior_counts = prior_counts

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
        if self.n_states is None:
            self.n_states = np.max([np.max(x) for x in sequences]) + 1

        from msmbuilder import MSMLib
        MSMLib.logger.info = lambda *args : None
        from msmbuilder.msm_analysis import get_eigenvectors
        from msmbuilder.MSMLib import mle_reversible_count_matrix, estimate_transition_matrix, ergodic_trim

        self.rawcounts_ = self._count_transitions(sequences)
        if self.prior_counts > 0:
            self.rawcounts_ = scipy.sparse.csr_matrix(self.rawcounts_.todense() + self.prior_counts)

        # STEP (1): Ergodic trimming
        if self.ergodic_trim:
            self.rawcounts_, mapping = ergodic_trim(scipy.sparse.csr_matrix(self.rawcounts_))
            self.mapping_ = {}
            for i, j in enumerate(mapping):
                if j != -1:
                    self.mapping_[i] = j
        else:
            self.mapping_ = dict((zip(np.arange(self.n_states), np.arange(self.n_states))))

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
            from_states = sequence[: -self.lag_time: 1]
            to_states = sequence[self.lag_time::1]
            transitions = np.row_stack((from_states, to_states))
            C = scipy.sparse.coo_matrix((np.ones(transitions.shape[1], dtype=int), transitions), shape=(self.n_states, self.n_states))
            counts = counts + C

        return counts

    @property
    def timescales_(self):
        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.transmat_.shape[0] - 3

        u, v = scipy.sparse.linalg.eigs(self.transmat_, k=n_timescales + 1)
        order = np.argsort(-np.real(u))
        u = np.real_if_close(u[order])

        # make sure to leave off equilibrium distribution
        timescales = - self.lag_time / np.log(u[1:])
        return timescales

