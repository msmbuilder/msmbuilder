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

from __future__ import print_function
import numpy as np
import scipy.linalg
import scipy.optimize

from ..base import BaseEstimator
from ..utils import list_of_1d
from . import _ratematrix
from .core import _MappingTransformMixin, _transition_counts


class ContinousTimeMSM(BaseEstimator, _MappingTransformMixin):
    """Reversible first order master equation model

    This model fits a reversible continuous-time Markov model for labeled
    sequence data.

    Parameters
    ----------
    lag_time : int
        The lag time of the model
    prior_counts : float, optional
        Add a number of "pseudo counts" to each entry in the counts matrix.
        When prior_counts == 0 (default), the assigned transition
        probability between two states with no observed transitions will be
        zero, whereas when prior_counts > 0, even this unobserved transitions
        will be given nonzero probability.
    verbose : bool
        Verbosity level

    Attributes
    ----------
    n_states_ : int
        The number of states
    ratemat_ : np.ndarray, shape=(n_states_, n_state_)
    transmat_ : np.ndarray, shape=(n_states_, n_state_)
    countsmat_ : array_like, shape = (n_states_, n_states_)
        Number of transition counts between states. countsmat_[i, j] is counted
        during `fit()`. The indices `i` and `j` are the "internal" indices
        described above. No correction for reversibility is made to this
        matrix.
    optimizer_state_ : object
    mapping_ : dict
        Mapping between "input" labels and internal state indices used by the
        counts and transition matrix for this Markov state model. Input states
        need not necessarily be integers in (0, ..., n_states_ - 1), for
        example. The semantics of ``mapping_[i] = j`` is that state ``i`` from
        the "input space" is represented by the index ``j`` in this MSM.
    """
    def __init__(self, lag_time=1, prior_counts=0, verbose=True):
        self.lag_time = lag_time
        self.prior_counts = prior_counts
        self.verbose = verbose

        self.ratemat_ = None
        self.transmat_ = None
        self.countsmat_ = None
        self.n_states_ = None
        self.optimizer_state_ = None
        self.mapping_ = None

    def fit(self, sequences, y=None):
        sequences = list_of_1d(sequences)
        countsmat, mapping = _transition_counts(sequences, self.lag_time)

        n_states = countsmat.shape[0]
        result = self._optimize(countsmat)

        K = np.zeros((n_states, n_states))
        _ratematrix.buildK(np.exp(result.x), n_states, K)

        self.ratemat_ = K
        self.transmat_ = scipy.linalg.expm(K)
        self.countsmat_ = countsmat
        self.n_states_ = n_states
        self.optimizer_state_ = result
        self.mapping_ = mapping

        print(self.optimizer_state_.message)
        print('ratemat\n', self.ratemat_)
        print('transmat\n', self.transmat_)

        return self

    def _optimize(self, countsmat):
        n = countsmat.shape[0]

        def objective(theta):
            f, g = _ratematrix.loglikelihood(theta, countsmat, n)
            return -f, -g

        theta0 = self.initial_guess(countsmat)
        result = scipy.optimize.minimize(
            fun=objective, x0=theta0, method='L-BFGS-B', jac=True,
            options={'disp': self.verbose})

        return result

    def initial_guess(self, countsmat):
        C = 0.5 * (countsmat + countsmat.T) + self.prior_counts
        pi = C.sum(axis=0) / C.sum(dtype=float)
        transmat = C.astype(float) / C.sum(axis=1)[:, None]

        K = np.real(scipy.linalg.logm(transmat))
        S = np.multiply(np.sqrt(np.outer(pi, 1/pi)), K)
        sflat = np.maximum(S[np.triu_indices_from(countsmat, k=1)], 1e-10)
        theta = np.concatenate((np.log(sflat), np.log(pi)))

        return theta
