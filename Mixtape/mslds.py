#import sys
import numpy as np
#from numpy import dot, shape, eye, outer, sum, log, zeros
#from numpy import nonzero, reshape, diag, copy, ones
#from numpy.linalg import svd, inv, eig
from numpy.random import rand, randn, multinomial, multivariate_normal
#from mixtape.utils import *
#from mixtape.A_sdp import *
#from mixtape.Q_sdp import *
import scipy.linalg as linalg
#import scipy.stats as stats


from mdtraj.utils import ensure_type
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
    """
    def __init__(self, n_states, n_features, As=None, bs=None, Qs=None, transmat=None):
        As = ensure_type(As, np.double, ndim=3, name='As', shape=(n_states, n_features, n_features), can_be_none=True)
        bs = ensure_type(bs, np.double, ndim=2, name='bs', shape=(n_states, n_features), can_be_none=True)
        Qs = ensure_type(Qs, np.double, ndim=3, name='Qs', shape=(n_states, n_features, n_features), can_be_none=True)
        transmat = ensure_type(transmat, np.double, ndim=3, name='transmat', shape=(n_states, n_states), can_be_none=True)

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
        if Qs is None:
            Qs = np.empty((n_states, n_features, n_features))
            for i in range(n_states):
                r = rand(n_features, n_features)
                r = (1.0 / n_features) * np.dot(r.T, r)
                Qs[i] = r
        if transmat is None:
            transmat = rand(n_states, n_states)
            transmat = self.Z / (sum(self.Z, axis=0))

        self.n_states = n_states
        self.n_features = n_features
        self.As_ = As
        self.bs_ = bs
        self.Qs_ = Qs
        self.transmat_ = transmat

    def sample(self, T, s_init=None, x_init=None, y_init=None):
        """
        Inputs:
          T: time to run simulation
        Outputs:
          xs: Hidden continuous states
          Ss: Hidden switch states
        """
        x_dim, y_dim, = self.x_dim, self.y_dim
        # Allocate Memory
        xs = zeros((T, x_dim))
        Ss = zeros(T)
        # Compute Invariant
        _, vl = linalg.eig(self.Z, left=True, right=False)
        pi = vl[:, 0]
        # Sample Start conditions
        sample = multinomial(1, pi, size=1)
        if s_init == None:
            Ss[0] = nonzero(sample)[0][0]
        else:
            Ss[0] = s_init
        if x_init == None:
            xs[0] = multivariate_normal(self.mus[Ss[0]], self.Sigmas[Ss[0]])
        else:
            xs[0] = x_init
        # Perform time updates
        for t in range(0, T - 1):
            s = Ss[t]
            A = self.As[s]
            b = self.bs[s]
            Q = self.Qs[s]
            xs[t + 1] = multivariate_normal(dot(A, xs[t]) + b, Q)
            sample = multinomial(1, self.Z[s], size=1)[0]
            Ss[t + 1] = nonzero(sample)[0][0]
        return (xs, Ss)

    def Viterbi(self, xs):
        """
        Identify the most likely hidden state sequence.
        """
        K, mus, Sigmas, Z = self.K, self.mus, self.Sigmas, self.Z
        (T, xdim) = shape(xs)
        logVs = zeros((T, K))
        Ptr = zeros((T, K))
        assignment = zeros(T)
        for k in range(K):
            logVs[0, k] = log_multivariate_normal_pdf(xs[0], mus[k], Sigmas[k])
            Ptr[0, k] = k
        for t in range(T - 1):
            for k in range(K):
                logVs[t + 1, k] = log_multivariate_normal_pdf(xs[t + 1], mus[k],
                                                              Sigmas[k]) + max(log(Z[:, k]) + logVs[t])
                Ptr[t + 1, k] = argmax(log(Z[:, k]) + logVs[t])
        assignment[T - 1] = argmax(logVs[T - 1, k])
        for t in range(T - 1, 0, -1):
            assignment[t - 1] = Ptr[t, assignment[t]]
        return assignment

    def em(self, ys, em_iters=10, em_vars='all'):
        """
        Expectation Maximization
        Inputs:
          ys: A sequence of shape (T, y_dim)
        Outputs:
          None (all updates are made to internal states)
        """
        (T, _) = shape(ys)
        K, x_dim, y_dim = self.K, self.x_dim, self.y_dim

        # regularization
        alpha = 0.1
        itr = 0
        W_i_Ts = zeros((em_iters, T, K))
        assignments = self.Viterbi(ys)
        while itr < em_iters:
            assignments = self.Viterbi(ys)
            W_i_T = assignment_to_weights(assignments, K)
            Zhat = transition_counts(assignments, K)
            M_tt_1T = tile(Zhat, (T, 1, 1))
            self.em_update(W_i_T, M_tt_1T,
                           ys, ys, alpha, itr, em_vars)
            W_i_Ts[itr] = W_i_T
            itr += 1

    def compute_sufficient_statistics(self, W_i_T, M_tt_1T, xs):
        (T, x_dim) = shape(xs)
        K = self.K

        stats = {}
        stats['cor'] = zeros((K, x_dim, x_dim))
        stats['cov'] = zeros((K, x_dim, x_dim))
        stats['cov_but_first'] = zeros((K, x_dim, x_dim))
        stats['cov_but_last'] = zeros((K, x_dim, x_dim))
        stats['mean'] = zeros((K, x_dim))
        stats['mean_but_first'] = zeros((K, x_dim))
        stats['mean_but_last'] = zeros((K, x_dim))
        stats['transitions'] = zeros((K, K))
        # Use Laplacian Pseudocounts
        stats['total'] = ones(K)
        stats['total_but_last'] = ones(K)
        stats['total_but_first'] = ones(K)
        for t in range(T):
            for k in range(K):
                if t > 0:
                    stats['cor'][k] += W_i_T[t, k] * outer(xs[t], xs[t - 1])
                    stats['cov_but_first'][
                        k] += W_i_T[t, k] * outer(xs[t], xs[t])
                    stats['total_but_first'][k] += W_i_T[t, k]
                    stats['mean_but_first'][k] += W_i_T[t, k] * xs[t]
                stats['cov'][k] += W_i_T[t, k] * outer(xs[t], xs[t])
                stats['mean'][k] += W_i_T[t, k] * xs[t]
                stats['total'][k] += W_i_T[t, k]
                if t < T:
                    stats['total_but_last'][k] += W_i_T[t, k]
                    stats['mean_but_last'][k] += W_i_T[t, k] * xs[t]
                    stats['cov_but_last'][
                        k] += W_i_T[t, k] * outer(xs[t], xs[t])
        stats['transitions'] = M_tt_1T[0]
        return stats

    def em_update(self, W_i_T, M_tt_1T, xs, ys, alpha, itr,
                  em_vars='all'):
        """
        Inputs:
          W_i_T= P[S_{t}=i|x_{1:T}]
          M_tt_1Ts = \sum_t P[S_t=j,S_{t+1}=k|x_{1:T}]
          em_vars: Variables to learn
        """
        K, x_dim = self.K, self.x_dim
        (T, _) = shape(xs)
        P_cur_prev = zeros((T, x_dim, x_dim))
        P_cur = zeros((T, x_dim, x_dim))
        means, covars = empirical_wells(xs, W_i_T)
        stats = self.compute_sufficient_statistics(W_i_T, M_tt_1T, xs)
        for t in range(T):
            if t > 0:
                P_cur_prev[t] = outer(xs[t], xs[t - 1])
            P_cur[t] = outer(xs[t], xs[t])
        # Update Sigmas
        if 'Sigmas' in em_vars:
            self.Sigma_update(stats)
        # Update mus
        if 'mus' in em_vars:
            self.mu_update(stats)
        # Update Z
        if 'Z' in em_vars:
            self.Z_update(stats, T)
        if itr > 2:
            # Update Qs
            if 'Qs' in em_vars:
                self.Q_update(stats, covars)
            # Update As
            if 'As' in em_vars:
                self.A_update(stats, covars)
            # Update bs
            if 'bs' in em_vars:
                self.b_update(stats, means)

    def b_update(self, stats, means):
        K, x_dim = self.K, self.x_dim
        for i in range(K):
            mu = self.mus[i]
            #self.bs[i] = dot(eye(x_dim) - self.As[i], means[i])
            self.bs[i] = dot(eye(x_dim) - self.As[i], mu)

    def mu_update(self, stats):
        K, x_dim = self.K, self.x_dim
        for k in range(K):
            # Use Laplace Pseudocount
            self.mus[k] = stats['mean'][k] / (stats['total'][k])

    def Sigma_update(self, stats):
        K, x_dim = self.K, self.x_dim
        for k in range(K):
            mu = reshape(self.mus[k], (x_dim, 1))
            Sigma_num = (stats['cov'][k] +
                         -dot(mu, reshape(stats['mean'][k], (x_dim, 1)).T) +
                         -dot(reshape(stats['mean'][k], (x_dim, 1)), mu.T) +
                         stats['total'][k] * dot(mu, mu.T))
            Sigma_denom = stats['total'][k]
            self.Sigmas[k] = Sigma_num / Sigma_denom

    def Z_update(self, stats, T):
        K = self.K
        Z = zeros((K, K))
        for i in range(K):
            for j in range(K):
                Z[i, j] += (T - 1) * stats['transitions'][i, j]
                Z_denom = stats['total_but_last'][i]
                Z[i, j] /= Z_denom
        for i in range(K):
            s = sum(Z[i, :])
            Z[i, :] /= s
        self.Z = Z

    def A_update(self, stats, covars):
        K, x_dim = self.K, self.x_dim
        for i in range(K):
            b = reshape(self.bs[i], (x_dim, 1))
            B = stats['cor'][i]
            mean_but_last = reshape(stats['mean_but_last'][i], (x_dim, 1))
            C = dot(b, mean_but_last.T)
            E = stats['cov_but_last'][i]
            Sigma = self.Sigmas[i]
            Q = self.Qs[i]
            sol, _, G, _ = solve_A(x_dim, B, C, E, Sigma, Q)
            avec = array(sol['x'])
            avec = avec[1 + x_dim * (x_dim + 1) / 2:]
            A = reshape(avec, (x_dim, x_dim), order='F')
            self.As[i] = A

    def Q_update(self, stats, covars):
        K, x_dim = self.K, self.x_dim
        for i in range(self.K):
            A = self.As[i]
            Sigma = self.Sigmas[i]
            b = reshape(self.bs[i], (x_dim, 1))
            B = ((stats['cov_but_first'][i]
                  - dot(stats['cor'][i], A.T)
                  - dot(reshape(stats['mean_but_first'][i], (x_dim, 1)),
                        b.T))
                 + (-dot(A, stats['cor'][i].T) +
                    dot(A, dot(stats['cov_but_last'][i], A.T)) +
                    dot(A, dot(reshape(stats['mean_but_last'][i], (x_dim, 1)),
                               b.T)))
                 + (-dot(b, reshape(stats['mean_but_first'][i], (x_dim, 1)).T) +
                    dot(b, dot(reshape(stats['mean_but_last'][i], (x_dim, 1)).T,
                               A.T)) +
                    stats['total_but_first'][i] * dot(b, b.T)))
            sol, _, _, _ = solve_Q(x_dim, A, B, Sigma)
            qvec = array(sol['x'])
            qvec = qvec[1 + x_dim * (x_dim + 1) / 2:]
            Q = zeros((x_dim, x_dim))
            for j in range(x_dim):
                for k in range(j + 1):
                    vec_pos = j * (j + 1) / 2 + k
                    Q[j, k] = qvec[vec_pos]
                    Q[k, j] = Q[j, k]
            self.Qs[i] = Q

    def compute_metastable_wells(self):
        """Compute the metastable wells according to the formula
            x_i = (I - A)^{-1}b
          Output: wells
        """
        K, x_dim = self.K, self.x_dim
        wells = zeros((K, x_dim))
        for i in range(K):
            wells[i] = dot(inv(eye(x_dim) - self.As[i]), self.bs[i])
        return wells

    def compute_process_covariances(self):
        K, x_dim = self.K, self.x_dim
        covs = zeros((K, x_dim, x_dim))
        N = 10000
        for k in range(K):
            A = self.As[k]
            Q = self.Qs[k]
            V = iter_vars(A, Q, N)
            covs[k] = V
        return covs

    def compute_eigenspectra(self):
        K, x_dim = self.K, self.x_dim
        eigenspectra = zeros((K, x_dim, x_dim))
        for k in range(K):
            eigenspectra[k] = diag(eig(self.As[k])[0])
        return eigenspectra
