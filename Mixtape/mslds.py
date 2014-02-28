"""
An Implementation of the Metastable Switching LDS. A forward-backward
inference pass computes switch posteriors from the smoothed hidden states.
The switch posteriors are used in the M-step to update parameter estimates.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
# Author: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import numpy as np
from numpy.random import multivariate_normal, randn, rand
import scipy.linalg
import numpy.linalg
from sklearn import cluster
from sklearn.hmm import GaussianHMM
from sklearn.mixture import distribute_covar_matrix_to_match_covariance_type
from mdtraj.utils import ensure_type

from mixtape import _reversibility, _switching_var1
from mixtape._switching_var1 import SwitchingVAR1CPUImpl
from mixtape.mslds_solvers.mslds_A_sdp import solve_A
from mixtape.mslds_solvers.mslds_Q_sdp import solve_Q
from mixtape.utils import iter_vars, categorical

class MetastableSwitchingLDS(object):
    """Metastable Switching Linear Dynamical System, fit via maximum
    likelihood.

    This model is an extension of a hidden Markov model. In the HMM, when
    the system stays in a single metastable state, each sample is i.i.d
    from the state's output distribution. Instead, in this model, the
    within-state dynamics are modeled by a linear dynamical system. The
    overall process can be thought of as a Markov jump process between
    different linear dynamical systems.

    The evolution is governed by the following relations. :math:`S_t` is
    hidden discrete state in :math:`{1, 2,... K}` which evolves as a
    discrete-state Markov chain with K by K transition probability matrix
    `Z`. :math:`Y_t` is the observed signal, which evolves as a linear
    dynamical system with

    .. math::

        S_t ~ Categorical(Z_{s_{t-1}})
        Y_t = A_{s_t} Y_{t-1} + e(s_t)
        e(s_t) ~ \mathcal{N}(b_{S_t}, Q_{S_t})

    Attributes
    ----------


    Parameters
    ----------
    n_states : int
        The number of hidden states. Each state is characterized by a
        separate stable linear dynamical system that the process can jump
        between.
    n_features : int
        Dimensionality of the space.
    As : np.ndarray, shape=(n_states, n_features, n_features):
        Each `A[i]` is the LDS evolution operator for the system, conditional
        on it being in state `i`.
    bs : np.ndarray, shape=(n_states, n_features)
        Mean of the gaussian noise in each state.
    Qs : np.ndarray, shape=(n_states, n_features, n_features)
        Local Covariance matrix for the noise in each state
    covars : np.ndarray, shape=(n_states, n_features, n_features)
        Global Covariance matrix for the noise in each state
    transmat : np.ndarray, shape=(n_states, n_states)
        State-to-state Markov jump probabilities
    n_iter : int, optional
        Number of iterations to perform during training
    reversible_type : str
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization (BFGS) and is the only current
        option.
    init_params : string, optional, default
        Controls which parameters are initialized prior to training. Can
        contain any combination of 't' for transmat, 'm' for means, and
        'c' for covars, 'q' for Q matrices, 'a' for A matrices, and 'b'
        for b vectors. Defaults to all parameters.
    params : string, optional, default
        Controls which parameters are updated in the training process.
        Can contain any combination of 't' for transmat, 'm' for means,
        and 'c' for covars, 'q' for Q matrices, 'a' for A matrices, and
        'b' for b vectors. Defaults to all parameters.
    eps: float, optional
        The transition matrices A[i] are initialized as (1-eps)*I and local
        covariance matrices Q[i] are initialized as eps*covars[i]. eps
        encodes the fact that local covariances Q[i] should be small and
        that A[i] should almost be identity.
    """

    def __init__(self, n_states, n_features, n_hotstart_sequences=10,
        init_params='tmcqab', transmat_prior=None, params='tmcqab',
        reversible_type='mle', n_iter=10, covars_prior=1e-2,
        covars_weight=1, precision='mixed', eps=2.e-1, platform='cpu'):

        self.n_states = n_states
        self.n_features = n_features
        self.n_hotstart_sequences = n_hotstart_sequences
        self.n_iter = n_iter
        self.init_params = init_params
        self.platform = platform
        self.reversible_type = reversible_type
        self.transmat_prior = transmat_prior
        self.params = params
        self.precision = precision
        if covars_prior <= 0:
            covars_prior = 0
            covars_weight = 0
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.eps = eps
        self._impl = SwitchingVAR1CPUImpl(n_states, n_features, precision)

        self._As_ = None
        self._bs_ = None
        self._Qs_ = None
        self._covars_ = None
        self._means_ = None
        self._transmat_ = None
        self._populations_ = None

        if not reversible_type in ['mle']:
            raise ValueError('Invalid value for reversible_type: %s '
                             'Must be "mle"' % reversible_type)

        if self.transmat_prior is None:
            self.transmat_prior = 1.0
        if self.platform == 'cpu':
            self._impl = _switching_var1.SwitchingVAR1CPUImpl(
                            self.n_states, self.n_features, precision)
        else:
            raise ValueError(('Invalid platform "%s".'
                    + 'Available platforms are %s.') % (platform,
                        ', '.join(_AVAILABLE_PLATFORMS)))

    def _init(self, sequences):
        """Initialize the state, prior to fitting (hot starting)
        """
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s')
           for s in sequences]
        self._impl._sequences = sequences

        small_dataset = np.vstack(
            sequences[0:min(len(sequences), self.n_hotstart_sequences)])

        if 'm' in self.init_params:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.means_ = cluster.KMeans(n_clusters=self.n_states).fit(small_dataset).cluster_centers_
        if 'c' in self.init_params:
            cv = np.cov(small_dataset.T)
            self.covars_ = \
                distribute_covar_matrix_to_match_covariance_type(
                  cv, 'full', self.n_states)
            self.covars_[self._covars_==0] = 1e-5
        if 't' in self.init_params:
            transmat_ = np.empty((self.n_states, self.n_states))
            transmat_.fill(1.0 / self.n_states)
            self.transmat_ = transmat_
            self.populations_ = np.ones(self.n_states) / self.n_states
        if 'a' in self.init_params:
            self.As_ = np.zeros((self.n_states, self.n_features, self.n_features))
            for i in range(self.n_states):
                self.As_[i] = np.eye(self.n_features) - self.eps
        if 'b' in self.init_params:
            self.bs_ = np.zeros((self.n_states, self.n_features))
            for i in range(self.n_states):
                A = self.As_[i]
                mean = self.means_[i]
                self.bs_[i] = np.dot(np.eye(self.n_features) -A, mean)
        if 'q' in self.init_params:
            self.Qs_ = np.zeros((self.n_states, self.n_features,
                self.n_features))
            for i in range(self.n_states):
                self.Qs_[i] = self.eps * self.covars_[i]


    def sample(self, n_samples, init_state=None, init_obs=None):
        """Sample a trajectory from model distribution

        Parameters
        ----------
        n_samples : int
            The length of the trajectory
        init_state : int
            The initial hidden metastable state, in {0, ..., n_states-1}
        init_obs : np.ndarray, shape=(n_features)
            The initial "observed" data point.

        Returns
        -------
        obs : np.ndarray, shape=(n_samples, n_features)
            The "observed" data samples.
        hidden_state : np.ndarray, shape=(n_samples, n_states)
            The hidden state of the process.
        """
        # Allocate Memory
        obs = np.zeros((n_samples, self.n_features))
        hidden_state = np.zeros(n_samples, dtype=int)

        # set the initial values of the sequences
        if init_state is None:
            # Sample Start conditions
            hidden_state[0] = categorical(self.populations_)
        else:
            hidden_state[0] = init_state

        if init_obs is None:
            obs[0] = multivariate_normal(self.means_[hidden_state[0]],
                                         self.covars_[hidden_state[0]])
        else:
            obs[0] = init_obs

        # Perform time updates
        for t in range(n_samples - 1):
            s = hidden_state[t]
            A = self.As_[s]
            b = self.bs_[s]
            Q = self.Qs_[s]
            obs[t + 1] = multivariate_normal(np.dot(A, obs[t]) + b, Q)
            hidden_state[t + 1] = \
              categorical(self.transmat_[s])

        return obs, hidden_state

    def predict(self, obs):
        """Find most likely state sequence corresponding to `obs`.

        Parameters
        ----------
        obs : np.ndarray, shape=(n_samples, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        hidden_states : np.ndarray, shape=(n_states)
            Index of the most likely states for each observation
        """
        _, vl = scipy.linalg.eig(self.transmat_, left=True, right=False)
        startprob = vl[:, 0] / np.sum(vl[:, 0])

        model = GaussianHMM(n_components=self.n_states, covariance_type='full')
        model.startprob_ = startprob
        model.transmat_ = self.transmat_
        model.means_ = self.means_
        model.covars_ = self.covars_
        return model.predict(obs)

    def fit(self, sequences):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        self._init(sequences)
        n_obs = sum(len(s) for s in sequences)

        for i in range(self.n_iter):
            print "Iteration %d" % i
            _, stats = self._impl.do_estep()
            if stats['trans'].sum() > 10*n_obs:
                print('Number of transition counts', stats['trans'].sum())
                print('Total sequence length', n_obs)
                print("Numerical overflow detected. Try splitting your trajectories")
                print("into shorter segments or running in double")
                break

            # Maximization step
            self._do_mstep(stats, set(self.params))


        return self

    def _do_mstep(self, stats, params):
        if 'm' in params:
            self._means_update(stats)
        if 'c' in params:
            self._covars_update(stats)
        if 't' in params:
            self._transmat_update(stats)

        if 'a' in params:
            self._A_update(stats)
        if 'q' in params:
            self._Q_update(stats)
        if 'b' in params:
            self._b_update(stats)

    def _means_update(self, stats):
        self.means_ = (stats['obs']) / (stats['post'][:, np.newaxis])

    def _covars_update(self, stats):
        cvnum = np.empty((self.n_states, self.n_features, self.n_features))
        for c in range(self.n_states):
            obsmean = np.outer(stats['obs'][c], self._means_[c])

            cvnum[c] = (stats['obs*obs.T'][c]
                        - obsmean - obsmean.T
                        + np.outer(self._means_[c], self._means_[c])
                        * stats['post'][c]) \
                + self.covars_prior * np.eye(self.n_features)
        cvweight = max(self.covars_weight - self.n_features, 0)
        self.covars_ = ((cvnum) /
                 (cvweight + stats['post'][:, None, None]))

    def _transmat_update(self, stats):
        counts = np.maximum(stats['trans'] + self.transmat_prior - 1.0, 1e-20).astype(np.float64)
        self.transmat_, self.populations_ = _reversibility.reversible_transmat(counts)

    def _A_update(self, stats):
        for i in range(self.n_states):
            b = np.reshape(self.bs_[i], (self.n_features, 1))
            B = stats['obs*obs[t-1].T'][i]
            mean_but_last = np.reshape(stats['obs[:-1]'][i],
                    (self.n_features, 1))
            C = np.dot(b, mean_but_last.T)
            E = stats['obs[:-1]*obs[:-1].T'][i]
            Sigma = self.covars_[i]
            Q = self.Qs_[i]
            sol, _, G, _ = solve_A(self.n_features, B, C, E, Sigma, Q)
            avec = np.array(sol['x'])
            avec = avec[1 + self.n_features * (self.n_features + 1) / 2:]
            A = np.reshape(avec, (self.n_features, self.n_features),
                    order='F')
            self.As_[i] = A

    def _Q_update(self, stats):
        for i in range(self.n_states):
            A = self.As_[i]
            Sigma = self.covars_[i]
            b = np.reshape(self.bs_[i], (self.n_features, 1))
            B = ((stats['obs[1:]*obs[1:].T'][i]
                - np.dot(stats['obs*obs[t-1].T'][i], A.T)
                - np.dot(np.reshape(stats['obs[1:]'][i],
                    (self.n_features, 1)), b.T))
                + (-np.dot(A, stats['obs*obs[t-1].T'][i].T) +
                    np.dot(A, np.dot(stats['obs[:-1]*obs[:-1].T'][i],
                        A.T)) +
                    np.dot(A, np.dot(np.reshape(stats['obs[:-1]'][i],
                        (self.n_features, 1)), b.T)))
                 + (-np.dot(b, np.reshape(stats['obs[1:]'][i],
                        (self.n_features, 1)).T) +
                    np.dot(b, np.dot(np.reshape(stats['obs[:-1]'][i],
                        (self.n_features, 1)).T,
                               A.T)) +
                    stats['post[1:]'][i] * np.dot(b, b.T)))
            sol, _, _, _ = solve_Q(self.n_features, A, B, Sigma)
            qvec = np.array(sol['x'])
            qvec = qvec[1 + self.n_features * (self.n_features + 1) / 2:]
            Q = np.zeros((self.n_features, self.n_features))
            for j in range(self.n_features):
                for k in range(j + 1):
                    vec_pos = j * (j + 1) / 2 + k
                    Q[j, k] = qvec[vec_pos]
                    Q[k, j] = Q[j, k]
            self.Qs_[i] = Q

    def _b_update(self, stats):
        for i in range(self.n_states):
            mu = self.means_[i]
            self.bs_[i] = np.dot(np.eye(self.n_features) - self.As_[i], mu)

    @property
    def As_(self):
      return self._As_
    @As_.setter
    def As_(self,value):
      value = np.asarray(value, order='c', dtype=np.float32)
      self._As_ = value
      self._impl.As_ = value

    @property
    def Qs_(self):
      return self._Qs_
    @Qs_.setter
    def Qs_(self,value):
      value = np.asarray(value, order='c', dtype=np.float32)
      self._Qs_ = value
      self._impl.Qs_ = value

    @property
    def bs_(self):
      return self._bs_
    @bs_.setter
    def bs_(self,value):
      value = np.asarray(value, order='c', dtype=np.float32)
      self._bs_ = value
      self._impl.bs_ = value

    @property
    def means_(self):
      return self._means_
    @means_.setter
    def means_(self, value):
      value = np.asarray(value, order='c', dtype=np.float32)
      self._means_ = value
      self._impl.means_ = value

    @property
    def covars_(self):
        return self._covars_
    @covars_.setter
    def covars_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._covars_ = value
        self._impl.covars_ = value

    @property
    def transmat_(self):
        return self._transmat_
    @transmat_.setter
    def transmat_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._transmat_ = value
        self._impl.transmat_ = value

    @property
    def populations_(self):
        return self._populations_
    @populations_.setter
    def populations_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._populations_ = value
        self._impl.startprob_ = value


    def compute_metastable_wells(self):
        """Compute the metastable wells according to the formula
            x_i = (I - A)^{-1}b
          Output: wells
        """
        wells = np.zeros((self.n_states, self.n_features))
        for i in range(self.n_states):
            wells[i] = np.dot(np.linalg.inv(np.eye(self.n_features) -
                            self.As_[i]), self.bs_[i])
        return wells

    def compute_process_covariances(self, N=10000):
        """Compute the emergent complexity D_i of metastable state i by
          solving the fixed point equation Q_i + A_i D_i A_i.T = D_i
          for D_i
        """
        covs = np.zeros((self.n_states, self.n_features, self.n_features))
        for k in range(self.n_states):
            A = self.As_[k]
            Q = self.Qs_[k]
            V = iter_vars(A, Q, N)
            covs[k] = V
        return covs

    def compute_eigenspectra(self):
        eigenspectra = np.zeros((self.n_states, self.n_features, self.n_features))
        for k in range(self.n_states):
            eigenspectra[k] = np.diag(np.linalg.eigvals(self.As_[k]))
        return eigenspectra
