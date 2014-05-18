"""
An Implementation of the Metastable Switching LDS. A forward-backward
inference pass computes switch posteriors from the smoothed hidden states.
The switch posteriors are used in the M-step to update parameter estimates.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
from __future__ import print_function, division, absolute_import
import warnings
import numpy as np
from numpy.random import multivariate_normal
import scipy.linalg
import time
from sklearn import cluster
from sklearn.hmm import GaussianHMM
from sklearn.mixture import distribute_covar_matrix_to_match_covariance_type
from mdtraj.utils import ensure_type

from mixtape.mslds_solver import MetastableSwitchingLDSSolver
from mixtape._mslds import MetastableSLDSCPUImpl
from mixtape.utils import iter_vars, categorical

class MetastableSwitchingLDS(object):

    """
    The Metastable Switching Linear Dynamical System (mslds) models
    within-state dynamics of metastable states by a linear dynamical
    system. The model is a Markov jump process between different linear
    dynamical systems.

    Parameters
    ----------
    n_states : int
        The number of hidden states.
    n_experiments : int
        Number of time the EM algorithm will be run with different
        random seeds.
    n_features : int
        Dimensionality of the space.
    n_hotstart : {int}
        Number of EM iterations for HMM hot-starting of mslds learning.
    n_em_iter : int, optional
        Number of iterations to perform during training
    params : string, optional, default
        Controls which parameters are updated in the training process.
    backet: string
        Either FirstOpt or cvxopt
    """

    def __init__(self, n_states, n_features, n_experiments=5,
            n_hotstart_sequences=10, params='tmcqab', n_em_iter=10,
            n_hotstart = 5, backend='FirstOpt'):

        self.n_states = n_states
        self.n_experiments = n_experiments
        self.n_features = n_features
        self.n_hotstart = n_hotstart
        self.n_hotstart_sequences = n_hotstart_sequences
        self.n_em_iter = n_em_iter
        self.params = params
        self.eps = .2
        self._As_ = None
        self._bs_ = None
        self._Qs_ = None
        self._covars_ = None
        self._means_ = None
        self._transmat_ = None
        self._populations_ = None

        self.solver = MetastableSwitchingLDSSolver(self.n_states,
                self.n_features)
        self.inferrer = MetastableSLDSCPUImpl(self.n_states,
                self.n_features, precision='mixed')


    def _init(self, sequences):
        """Initialize the state, prior to fitting (hot starting)
        """
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s')
                     for s in sequences]
        self.inferrer._sequences = sequences

        small_dataset = np.vstack(
            sequences[0:min(len(sequences), self.n_hotstart_sequences)])

        # Initialize means
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.means_ = (cluster.KMeans(n_clusters=self.n_states)
                .fit(small_dataset).cluster_centers_)

        # Initialize covariances
        cv = np.cov(small_dataset.T)
        self.covars_ = \
            distribute_covar_matrix_to_match_covariance_type(
                cv, 'full', self.n_states)
        self.covars_[self.covars_ == 0] = 1e-5

        # Initialize transmat
        transmat_ = np.empty((self.n_states, self.n_states))
        transmat_.fill(1.0 / self.n_states)
        self.transmat_ = transmat_
        self.populations_ = np.ones(self.n_states) / self.n_states

        # Initialize As
        self.As_ = np.zeros((self.n_states, self.n_features,
            self.n_features))
        for i in range(self.n_states):
            self.As_[i] = np.eye(self.n_features) - self.eps

        # Initialize means
        self.bs_ = np.zeros((self.n_states, self.n_features))
        for i in range(self.n_states):
            A = self.As_[i]
            mean = self.means_[i]
            self.bs_[i] = np.dot(np.eye(self.n_features) - A, mean)

        # Initialize local covariances
        self.Qs_ = np.zeros((self.n_states, self.n_features,
                             self.n_features))
        for i in range(self.n_states):
            self.Qs_[i] = self.eps * self.covars_[i]

    def sample(self, n_samples, init_state=None, init_obs=None):
        """Sample a trajectory from model distribution
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
            hidden_state[t + 1] = categorical(self.transmat_[s])

        return obs, hidden_state

    def score(self, sequences):
        """Log-likelihood of sequences under the model
        """
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s')
                     for s in sequences]
        self.inferrer._sequences = sequences
        logprob, _ = self.inferrer.do_mslds_estep()
        return logprob

    def fit(self, sequences):
        """Estimate model parameters.
        """
        n_obs = sum(len(s) for s in sequences)
        best_fit = {'params': {}, 'loglikelihood': -np.inf}

        for r in range(self.n_experiments):
            fit_logprob = []
            self._init(sequences)
            for i in range(self.n_hotstart):
                curr_logprob, stats = self.inferrer.do_hmm_estep()
                self.means, self.covars = self.solver.do_hmm_mstep(stats)
            for i in range(self.n_em_iter):
                curr_logprob, stats = self.inferrer.do_mslds_estep()
                fit_logprob.append(curr_logprob)
                self.transmat_, self.As_, self.Qs_, self.bs_ \
                        = self.solver.do_mstep(self.As_, self.Qs_,
                                self.bs_, self.means_, self.covars_, stats)
                if curr_logprob > best_fit['loglikelihood']:
                    best_fit['loglikelihood'] = curr_logprob
                    best_fit['params'] = {'means': self.means_,
                                          'covars': self.covars_,
                                          'As': self.As_,
                                          'bs': self.bs_,
                                          'Qs': self.Qs_,
                                          'populations': self.populations_,
                                          'transmat': self.transmat_,
                                          'fit_logprob': fit_logprob}
        # Set the final values
        self.means_ = best_fit['params']['means']
        self.vars_ = best_fit['params']['covars']
        self.As_ = best_fit['params']['As']
        self.bs_ = best_fit['params']['bs']
        self.Qs_ = best_fit['params']['Qs']
        self.transmat_ = best_fit['params']['transmat']
        self.populations_ = best_fit['params']['populations']
        self.fit_logprob_ = best_fit['params']['fit_logprob']

        return self

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
          Can this be deleted?
        """
        covs = np.zeros((self.n_states, self.n_features, self.n_features))
        for k in range(self.n_states):
            A = self.As_[k]
            Q = self.Qs_[k]
            V = iter_vars(A, Q, N)
            covs[k] = V
        return covs

    # Boilerplate setters and getters
    @property
    def As_(self):
        return self._As_

    @As_.setter
    def As_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._As_ = value
        self.inferrer.As_ = value

    @property
    def Qs_(self):
        return self._Qs_

    @Qs_.setter
    def Qs_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._Qs_ = value
        self.inferrer.Qs_ = value

    @property
    def bs_(self):
        return self._bs_

    @bs_.setter
    def bs_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._bs_ = value
        self.inferrer.bs_ = value

    @property
    def means_(self):
        return self._means_

    @means_.setter
    def means_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._means_ = value
        self.inferrer.means_ = value

    @property
    def covars_(self):
        return self._covars_

    @covars_.setter
    def covars_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._covars_ = value
        self.inferrer.covars_ = value

    @property
    def transmat_(self):
        return self._transmat_

    @transmat_.setter
    def transmat_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._transmat_ = value
        self.inferrer.transmat_ = value

    @property
    def populations_(self):
        return self._populations_

    @populations_.setter
    def populations_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._populations_ = value
        self.inferrer.startprob_ = value
