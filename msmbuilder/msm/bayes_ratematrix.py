import time
import numpy as np
import scipy.misc

from msmbuilder.cluster import NDGrid
from msmbuilder.example_datasets import load_doublewell
from msmbuilder.msm import _ratematrix, ContinuousTimeMSM
from msmbuilder.msm.core import _normalize_eigensystem

from msmbuilder.msm._ratematrix import loglikelihood, ldirichlet_softmax
from msmbuilder.msm._ratematrix import lexponential

from msmbuilder.utils import experimental
from msmbuilder.base import BaseEstimator
from msmbuilder.msm.core import _MappingTransformMixin

def _log_posterior(theta, counts, alpha, beta, n, inds=None):
    """

    """
    # likelihood + grad
    logp1, grad = loglikelihood(theta, counts, n=n, inds=inds)

    # exponential prior on s_{ij}
    logp2 = lexponential(theta[:-n], beta, grad=grad[:-n])

    # dirichlet prior on \pi
    logp3 = ldirichlet_softmax(theta[-n:], alpha=alpha, grad=grad[-n:])

    logp = logp1 + logp2 + logp3
    return logp, grad


class HMCContinuousTimeMSM(BaseEstimator, _MappingTransformMixin):
    def __init__(self, lag_time=1,  n_samples=1000, n_steps=25, epsilon=0.01,
                 prior_alpha=1, prior_beta=1, n_timescales=None,
                 use_sparse=True, sliding_window=True, verbose=False):
        self.lag_time = lag_time
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_timescales = n_timescales
        self.use_sparse = use_sparse
        self.sliding_window = sliding_window
        self.verbose = verbose

    @experimental('HMCContinuousTimeMSM')
    def fit(self, sequences, y=None):
        model = ContinuousTimeMSM(
            lag_time=self.lag_time, prior_counts=0,
            n_timescales=self.n_timescales, use_sparse=self.use_sparse,
            sliding_window=self.sliding_window, verbose=self.verbose)
        model.fit(sequences)
        self.countsmat_ = model.countsmat_
        self.n_states_ = model.n_states_
        self.theta0_ = model.theta_
        self.inds_ = model.inds_

        samples, diag = self.sample()
        return self

    def sample(self):
        from pyhmc import hmc

        alpha = self.prior_alpha
        beta = self.prior_beta
        if np.isscalar(self.prior_alpha):
            # symmetric dirichlet
            alpha = self.prior_alpha * np.ones(self.n_states_)
        if np.isscalar(self.prior_beta):
            beta = self.prior_beta * np.ones(len(self.theta0_) - self.n_states_)

        def func(theta):
            logp, grad = _log_posterior(theta, self.countsmat_,
                alpha=alpha, beta=beta, n=self.n_states_, inds=self.inds_)
            return logp, grad

        samples, diag = hmc(func, x0=self.theta0_, n_samples=self.n_samples,
                            epsilon=self.epsilon, n_steps=self.n_steps,
                            return_diagnostics=True)
        return samples, diag



def _solve_eigensystem(theta, n, inds):
    S = np.zeros((n, n))
    exptheta = np.exp(theta)
    _ratematrix.build_ratemat(exptheta, n, inds, S, which='S')
    u, lv, rv = map(np.asarray, _ratematrix.eig_K(S, n, exptheta[-n:], 'S'))
    order = np.argsort(-u)
    u = u[order]
    lv = lv[:, order]
    rv = rv[:, order]

    return _normalize_eigensystem(u, lv, rv)


def main():
    n = 4
    grid = NDGrid(n_bins_per_feature=n, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])
    model = ContinuousTimeMSM(use_sparse=True).fit(seqs)
    print(model.timescales_)

    HMCContinuousTimeMSM().fit(seqs)

    # calls = [0]
    # def func(theta):
    #     calls[0] += 1
    #     start = time.time()
    #     logp, grad = log_posterior(theta, model.countsmat_, alpha, beta, n, inds=model.inds_)
    #     return logp, grad
    #
    # samples, diag = hmc(func, x0=model.theta_, n_samples=1000, epsilon=0.02,
    #                     n_steps=25, return_diagnostics=True)
    # print('rejection rate', diag['rej'])
    #
    # ts = []
    # for i in range(samples.shape[0]):
    #     ts.append(-1 / _solve_eigensystem(samples[i], n, inds=model.inds_)[0][1])
    # ts = np.array(ts)
    #
    # import matplotlib.pyplot as pp
    # pp.hist(ts, bins=25)
    # pp.savefig('Pyplots.pdf')


if __name__ == '__main__':
    main()
