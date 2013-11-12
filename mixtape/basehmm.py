"""
Base class for reversible hidden Markov models.

Reversibility ensures that the model has a unique stationary
distribution and that the transition matrix has only real
eigenvectors. Such a model satisfies the detailed balance
condition -- that an equilibrium state of the chain exists
wherein the flux in and out of each hidden states are identical.
This means that there are no 'sink' or 'source' states.

The starting probabilities of the chain over the hidden states
are free parameters in this model. Instead, the initial distribution
of the hidden state in any observation sequence is assumed to come
from the chain's stationary distribution.
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2013, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division
import string
import numpy as np
from sklearn.utils.extmath import logsumexp
import sklearn._hmmc
from sklearn.hmm import _BaseHMM, normalize
import mixtape._hmm

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class _ReversibleHMM(_BaseHMM):
    """Base class for reversible hidden Markov models.

    Reversibility ensures that the model has a unique stationary
    distribution and that the transition matrix has only real
    eigenvectors. Such a model satisfies the detailed balance
    condition -- that an equilibrium state of the chain exists
    wherein the flux in and out of each hidden states are identical.
    This means that there are no 'sink' or 'source' states.

    The starting probabilities of the chain over the hidden states
    are free parameters in this model. Instead, the initial distribution
    of the hidden state in any observation sequence is assumed to come
    from the chain's stationary distribution.

    Parameters
    ----------
    n_components : int
        The number of components (states) in the model
    n_iter : int
        The number of iterations of expectation-maximization to run
    thresh : float
        Convergence threshold for the log-likelihodo during expectation
        maximization. When the increase in the log-likelihood is less
        than thresh between subsequent rounds of E-M, fitting will finish.
    reversible_type : str
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization (BFGS), and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    params : str
        A string with the parameters to optimizing during the fitting.
        If 't' is in params, the transition matrix will be optimized.
        Subclasses may include their own parameters.
    init_params : str
        A string with the parameters to initialize prior to fitting.
        If 't' is in init_params, the transition matrix will be initialized
        prior to fitting.

    Attributes
    ----------
    fit_logprob_ : list
        The log-likelihood of the model at each step of E-M in the last round
        of fitting.
    populations_ : array, shape=[n_components]
        The equilibrium population of each component, given by the stationary
        eigenvector of the transition matrix. These populations are also used
        as the starting probabilities of each chain, which assumes that the
        chains are sampled from an equilibrium distribution. Unlike other
        HMM implementations, the starting probabilities are not free parameters
        here.
    transmat_
        The transition matrix. transmat_[i,j] gives the probability of the hidden
        state transitioning from state i to state j in a single step.
    """

    def __init__(self, n_components=1, n_iter=100, thresh=1e-2,
                 reversible_type='mle', transmat_prior=None,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        self.n_components = n_components
        self.n_iter = n_iter
        self.thresh = thresh
        self.transmat_prior = transmat_prior
        self.params = params
        self.init_params = init_params
        self.reversible_type = reversible_type

        if not reversible_type in ['mle', 'transpose']:
            raise ValueError('Invalid value for reversible_type: %s '
                             'Must be either "mle" or "transpose"'
                             % reversible_type)

    def fit(self, obs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        self._init()

        self.fit_logprob_ = []
        for i in range(self.n_iter):
            # Expectation step
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for seq in obs:
                framelogprob = self._compute_log_likelihood(seq)
                lpr, fwdlattice = self._do_forward_pass(framelogprob)
                bwdlattice = self._do_backward_pass(framelogprob)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
                curr_logprob += lpr
                self._accumulate_sufficient_statistics(
                    stats, seq, framelogprob, posteriors, fwdlattice,
                    bwdlattice, self.params)
            self.fit_logprob_.append(curr_logprob)

            # Check for convergence
            if i > 0 and abs(self.fit_logprob_[-1] - self.fit_logprob_[-2]) < self.thresh:
                break

            # Maximization step
            self._do_mstep(stats, self.params)
        return self

    def _init(self):
        if 't' in self.init_params:
            transmat_ = np.empty((self.n_components, self.n_components))
            transmat_.fill(1.0 / self.n_components)
            self.transmat_ = transmat_

    def _do_mstep(self, stats, params):
        if self.transmat_prior is None:
            self.transmat_prior = 1.0

        if 't' in params:
            if self.reversible_type == 'mle':
                counts = np.maximum(stats['trans'] + self.transmat_prior - 1.0, 1e-20)
                self.transmat_, self.populations_ = mixtape._hmm.reversible_transmat(counts)
                self._log_startprob = np.log(self.populations_)
            elif self.reversible_type == 'transpose':
                revcounts = np.maximum(self.transmat_prior - 1.0 + stats['trans'] + stats['trans'].T, 1e-20)
                self.populations_ = np.sum(revcounts, axis=0)
                self.transmat_ = normalize(revcounts, axis=1)
                self._log_startprob = np.log(self.populations_)
            else:
                raise ValueError('Invalid value for reversible_type: %s '
                                 'Must be either "mle" or "transpose"'
                                 % self.reversible_type)

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        stats['nobs'] += 1
        if 't' not in params or len(framelogprob) <= 1:
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            return

        n_observations, n_components = framelogprob.shape
        lneta = np.zeros((n_observations - 1, n_components, n_components))
        lnP = logsumexp(fwdlattice[-1])
        sklearn._hmmc._compute_lneta(n_observations, n_components, fwdlattice,
                             self._log_transmat, bwdlattice, framelogprob, lnP, lneta)
        stats["trans"] += np.exp(logsumexp(lneta, 0))

    def timescales_(self):
        """The implied relaxation timescales of the hidden Markov transition
        matrix

        By diagonalizing the transition matrix, its propagation of an arbitrary
        initial probability vector can be written as a sum of the eigenvectors
        of the transition weighted by per-eigenvector term that decays
        exponentially with time. Each of these eigenvectors describes a
        "dynamical mode" of the transition matrix and has a characteristic
        timescale, which gives the timescale on which that mode decays towards
        equilibrium. These timescales are given by :math:`-1/log(u_i)` where
        :math:`u_i` are the eigenvalues of the transition matrix. In an HMM
        with N components, the number of non-infinite timescales is N-1. (The
        -1 comes from the fact that the stationary distribution of the chain
        is associated with an eigenvalue of 1, and an infinite characteritic
        timescale).

        Returns
        -------
        timescales : array, shape=[n_components-1]
            The characteristic timescales of the transition matrix. If the model
            has not been fit or does not have a transition matrix, the return
            value will be None.
        """
        if not hasattr(self, 'transmat_'):
            return None
        eigvals = np.linalg.eigvals(self.transmat_)
        np.sort(eigvals)
        return -1 / np.log(eigvals[:-1])
