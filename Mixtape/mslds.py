#import sys
import numpy as np
from numpy.random import rand, randn, multivariate_normal
import scipy.linalg as linalg
from mdtraj.utils import ensure_type
from sklearn.hmm import GaussianHMM

from mixtape.A_sdp import solve_A
from mixtape.Q_sdp import solve_Q
from mixtape.utils import (iter_vars, categorical, transition_counts,
                           empirical_wells, assignment_to_weights)

"""
An Implementation of the Metastable Switching LDS. A forward-backward
inference pass computes switch posteriors from the smoothed hidden states.
The switch posteriors are used in the M-step to update parameter estimates.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""


class MetastableSwitchingLDS(object):
    """Metastable Switching Linear Dynamical System, fit via maximum likelihood.

    This model is an extension of a hidden Markov model. In the HMM, when the
    system stays in a single metastable state, each sample is i.i.d from the
    state's output distribution. Instead, in this model, the within-state
    dynamics are modeled by a linear dynamical system. The overall process can
    be thought of as a Markov jump process between different linear dynamical
    systems.

    The evolution is governed by the following relations. :math:`S_t` is hidden
    discrete state in :math:`{1, 2,... K}` which evolves as a discrete-state
    Markov chain with K by K transition probability matrix `Z`. :math:`Y_t` is
    the observed signal, which evolves as a linear dynamical system with

    .. math::

        S_t ~ Categorical(Z_{s_{t-1}})
        Y_t = A_{s_t} Y_{t-1} + e(s_t)
        e(s_t) ~ \mathcal{N}(b_{S_t}, Q_{S_t})

    Attributes
    ----------


    Parameters
    ----------
    n_states : int
        The number of hidden states. Each state is characterized by a separate
        stable linear dynamical system that the process can jump between.
    n_features : int
        Dimensionality of the space.
    As : np.ndarray, shape=(n_states, n_features, n_features):
        Each `A[i]` is the LDS evolution operator for the system, conditional
        on it being in state `i`.
    bs : np.ndarray, shape=(n_states, n_features)
        Mean of the gaussian noise in each state.
    Qs : np.ndarray, shape=(n_states, n_features, n_features)
        Covariance matrix for the noise in each state
    transmat : np.ndarray, shape=(n_states, n_states)
        State-to-state Markov jump probabilities
    n_iter : int, optional
        Number of iterations to perform during training
    params : string, optional, default
        Controls which parameters are updated in the training
        process.  Can contain any combination of
        't' for transmat, 'm' for means, and 's' for sigmas, 'q' for Q matrices,
        'a' for A matrices, and 'b' for b vectors. Defaults to all parameters.
    """
            
    def __init__(self, n_states, n_features, means=None, sigmas=None, As=None,
                 bs=None, Qs=None, transmat=None, n_iter=10, params='tmsqab'):
        As = ensure_type(As, np.double, ndim=3, name='As', shape=(n_states, n_features, n_features), can_be_none=True)
        bs = ensure_type(bs, np.double, ndim=2, name='bs', shape=(n_states, n_features), can_be_none=True)
        Qs = ensure_type(Qs, np.double, ndim=3, name='Qs', shape=(n_states, n_features, n_features), can_be_none=True)
        sigmas = ensure_type(sigmas, np.double, ndim=3, name='sigmas', shape=(n_states, n_features, n_features), can_be_none=True)
        means = ensure_type(means, np.double, ndim=2, name='means', shape=(n_states, n_features), can_be_none=True)
        transmat = ensure_type(transmat, np.double, ndim=3, name='transmat', shape=(n_states, n_states), can_be_none=True)

        # set default (random) values for the parameters before fitting, if not
        # supplied
        if As is None:
            # Produce a random As
            As = np.empty((n_states, n_features, n_features))
            for i in range(n_states):
                A = randn(n_features, n_features)
                # Stabilize A
                u, s, v = np.linalg.svd(A)
                As[i] = rand() * np.dot(u, v.T)

        if bs is None:
            bs = rand(n_states, n_features)
        if means is None:
            means = rand(n_states, n_features)

        if Qs is None:
            Qs = np.empty((n_states, n_features, n_features))
            for i in range(n_states):
                r = rand(n_features, n_features)
                r = (1.0 / n_features) * np.dot(r.T, r)
                Qs[i] = r

        if transmat is None:
            transmat = rand(n_states, n_states)
            transmat = self.Z / (sum(self.Z, axis=0))

        if sigmas is None:
            sigmas = np.empty((n_states, n_features, n_features))
            for i in range(n_states):
                r = rand(n_features, n_features)
                r = np.dot(r, r.T)
                sigmas[i] = 0.1 * np.eye(n_features) + r

        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter
        self.params = params
        self.As_ = As
        self.bs_ = bs
        self.Qs_ = Qs
        self.sigmas_ = sigmas
        self.means_ = means
        self.transmat_ = transmat

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
            # Compute the stationary distribution of the transition matrix
            _, vl = linalg.eig(self.transmat_, left=True, right=False)
            pi = vl[:, 0] / np.sum(vl[:, 0])
            # Sample Start conditions
            hidden_state[0] = categorical(pi)
        else:
            hidden_state[0] = init_state

        if init_obs is None:
            obs[0] = multivariate_normal(self.means_[hidden_state[0]],
                                         self.sigmas_[hidden_state[0]])
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
        _, vl = linalg.eig(self.transmat_, left=True, right=False)
        startprob = vl[:, 0] / np.sum(vl[:, 0])
        
        model = GaussianHMM(n_components=self.n_states, covariance_type='full')
        model.startprob_ = startprob
        model.transmat_ = self.transmat_
        model.means_ = self.means_
        model.covars_ = self.sigmas_
        return model.predict(obs)

    def fit(self, obs):
        """Estimate model parameters.
        
        Parameters
        ----------
        obs : np.ndarray, shape=(n_samples, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.
        """
        obs = ensure_type(obs, dtype=np.double, ndim=2, shape=(None, self.n_features))
        n_samples = len(obs)

        W_i_Ts = np.zeros((self.n_iter, n_samples, self.n_states))
        
        for i in range(self.n_iter):
            assignments = self.predict(obs)
            W_i_T = assignment_to_weights(assignments, self.n_states)
            Zhat = transition_counts(assignments, self.n_states)
            M_tt_1T = np.tile(Zhat, (n_samples, 1, 1))
            self._em_update(W_i_T, M_tt_1T, obs, i)
            W_i_Ts[i] = W_i_T

    def _initialize_sufficient_statistics(self):
        stats = {}
        stats['cor'] = np.zeros((self.n_states, self.n_features, self.n_features))
        stats['cov'] = np.zeros((self.n_states, self.n_features, self.n_features))
        stats['cov_but_first'] = np.zeros((self.n_states, self.n_features, self.n_features))
        stats['cov_but_last'] = np.zeros((self.n_states, self.n_features, self.n_features))
        stats['mean'] = np.zeros((self.n_states, self.n_features))
        stats['mean_but_first'] = np.zeros((self.n_states, self.n_features))
        stats['mean_but_last'] = np.zeros((self.n_states, self.n_features))
        stats['transitions'] = np.zeros((self.n_states, self.n_states))

        # Use Laplacian Pseudocounts
        stats['total'] = np.ones(self.n_states)
        stats['total_but_last'] = np.ones(self.n_states)
        stats['total_but_first'] = np.ones(self.n_states)

        return stats

    def compute_sufficient_statistics(self, W_i_T, M_tt_1T, obs):
        # TODO: refactor this so that we can compute sufficient statistics
        # over multiple trajectories

        stats = self._initialize_sufficient_statistics()
        n_samples = len(obs)
        for t in range(n_samples):
            for k in range(self.n_states):
                if t > 0:
                    stats['cor'][k] += W_i_T[t, k] * np.outer(obs[t], obs[t - 1])
                    stats['cov_but_first'][k] += W_i_T[t, k] * np.outer(obs[t], obs[t])
                    stats['total_but_first'][k] += W_i_T[t, k]
                    stats['mean_but_first'][k] += W_i_T[t, k] * obs[t]
                stats['cov'][k] += W_i_T[t, k] * np.outer(obs[t], obs[t])
                stats['mean'][k] += W_i_T[t, k] * obs[t]
                stats['total'][k] += W_i_T[t, k]
                if t < n_samples:
                    stats['total_but_last'][k] += W_i_T[t, k]
                    stats['mean_but_last'][k] += W_i_T[t, k] * obs[t]
                    stats['cov_but_last'][
                        k] += W_i_T[t, k] * np.outer(obs[t], obs[t])
        stats['transitions'] = M_tt_1T[0]
        return stats

    def _em_update(self, W_i_T, M_tt_1T, obs, itr):
        """
        
        """
        n_samples = len(obs)
        stats = self.compute_sufficient_statistics(W_i_T, M_tt_1T, obs)

        if 's' in self.params:
            self._sigmas_update(stats)
        if 'm' in self.params:
            self._means_update(stats)
        if 't' in self.params:
            self._transmat_update(stats, n_samples)

        if itr > 2:
            means, covars = empirical_wells(obs, W_i_T)

            if 'q' in self.params:
                self._Q_update(stats, covars)
            if 'a' in self.params:
                self._A_update(stats, covars)
            if 'b' in self.params:
                self._b_update(stats, means)

    def _b_update(self, stats, means):
        for i in range(self.n_states):
            mu = self.means_[i]
            self.bs_[i] = np.dot(np.eye(self.n_features) - self.As_[i], mu)

    def _means_update(self, stats):
        for i in range(self.n_states):
            self.means_[i] = stats['mean'][i] / stats['total'][i]

    def _sigmas_update(self, stats):
        for i in range(self.n_states):
            mu = np.reshape(self.means_[i], (self.n_features, 1))
            sigma_num = (stats['cov'][i] +
                         -np.dot(mu, np.reshape(stats['mean'][i], (self.n_features, 1)).T) +
                         -np.dot(np.reshape(stats['mean'][i], (self.n_features, 1)), mu.T) +
                         stats['total'][i] * np.dot(mu, mu.T))
            sigma_denom = stats['total'][i]
            self.sigmas_[i] = sigma_num / sigma_denom

    def _transmat_update(self, stats, T):
        Z = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                Z[i, j] += (T - 1) * stats['transitions'][i, j]
                Z_denom = stats['total_but_last'][i]
                Z[i, j] /= Z_denom
        for i in range(self.n_states):
            s = np.sum(Z[i, :])
            Z[i, :] /= s

        self.transmat_ = Z

    def _A_update(self, stats, covars):
        for i in range(self.n_states):
            b = np.reshape(self.bs_[i], (self.n_features, 1))
            B = stats['cor'][i]
            mean_but_last = np.reshape(stats['mean_but_last'][i], (self.n_features, 1))
            C = np.dot(b, mean_but_last.T)
            E = stats['cov_but_last'][i]
            Sigma = self.sigmas_[i]
            Q = self.Qs_[i]
            sol, _, G, _ = solve_A(self.n_features, B, C, E, Sigma, Q)
            avec = np.array(sol['x'])
            avec = avec[1 + self.n_features * (self.n_features + 1) / 2:]
            A = np.reshape(avec, (self.n_features, self.n_features), order='F')
            self.As_[i] = A

    def _Q_update(self, stats, covars):
        for i in range(self.n_states):
            A = self.As_[i]
            Sigma = self.sigmas_[i]
            b = np.reshape(self.bs_[i], (self.n_features, 1))
            B = ((stats['cov_but_first'][i]
                  - np.dot(stats['cor'][i], A.T)
                  - np.dot(np.reshape(stats['mean_but_first'][i], (self.n_features, 1)),
                        b.T))
                 + (-np.dot(A, stats['cor'][i].T) +
                    np.dot(A, np.dot(stats['cov_but_last'][i], A.T)) +
                    np.dot(A, np.dot(np.reshape(stats['mean_but_last'][i], (self.n_features, 1)),
                               b.T)))
                 + (-np.dot(b, np.reshape(stats['mean_but_first'][i], (self.n_features, 1)).T) +
                    np.dot(b, np.dot(np.reshape(stats['mean_but_last'][i], (self.n_features, 1)).T,
                               A.T)) +
                    stats['total_but_first'][i] * np.dot(b, b.T)))
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
