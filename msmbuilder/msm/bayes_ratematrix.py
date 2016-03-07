# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import
import time
import numpy as np

try:
    from pyhmc import hmc
    from pyhmc import integrated_autocorr2
    pyhmc_imported = True
except ImportError:
    pyhmc_imported = False

from .ratematrix import ContinuousTimeMSM
from .core import _solve_ratemat_eigensystem, _MappingTransformMixin
from ._ratematrix import loglikelihood, ldirichlet_softmax, lexponential
from ..utils import experimental
from ..base import BaseEstimator


class BayesianContinuousTimeMSM(BaseEstimator, _MappingTransformMixin):
    """Bayesian reversible first-order Master equation model.

    This model is a Bayesian variant of ``ContinuousTimeMSM`` which is sampled
    using Hamiltonian Monte Carlo (HMC). It requires the external package
    ``pyhmc``.

    .. warning::

        This model is currently experimental. It is *not* recommended for
        use in production calculations.

    Parameters
    ----------
    lag_time : int
        The lag time used to count the number of state to state transition
        events.
    n_samples : int
        Number of samples to generate from the HMC sampler.
    n_steps : int
        Trajectory length between Metropolized steps for the HMC integrator.
    epsilon : float
        Step size for the HMC integrator.
    prior_alpha : float > 0 or ndarray, shape=(n_states_,)
        Concentration parameter for a Dirichlet prior on the equilibrium
        distribution.
    prior_beta : float > 0 or ndarray, shape=(len(theta_)-n_states_,)
        Scale parameter for exponential prior on the symmetric rate matrix.
    n_timescales : int, optional
        Number of implied timescales to calculate.
    sliding_window : bool, optional
        Count transitions using a window of length ``lag_time``, which is slid
        along the sequences 1 unit at a time, yielding transitions which contain
        more data but cannot be assumed to be statistically independent. Otherwise,
        the sequences are simply subsampled at an interval of ``lag_time``.
    verbose : bool, default=False
        Verbosity level

    See Also
    --------
    ContinuousTimeMSM : maximum likelihood version

    References
    ----------
    .. [1] McGibbon, R. T. and Pande, V. S. "Parameterization of master
       equation models for chemical dynamics." In Preparation.

    Attributes
    ----------
    TODO(@rmcgibbo)
    """

    def __init__(self, lag_time=1,  n_samples=2000, n_steps=25, epsilon=5e-4,
                 prior_alpha=1, prior_beta=1, n_timescales=None,
                 sliding_window=True, verbose=False):
        self.lag_time = lag_time
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_timescales = n_timescales
        self.sliding_window = sliding_window
        self.verbose = verbose

        self._all_eigenvalues = None
        self._all_left_eigenvectors = None
        self._all_right_eigenvectors = None
        self._is_dirty = False

    @experimental('BayesianContinuousTimeMSM')
    def fit(self, sequences, y=None):
        model = ContinuousTimeMSM(
            lag_time=self.lag_time, n_timescales=self.n_timescales,
            sliding_window=self.sliding_window, verbose=self.verbose)
        model.fit(sequences)
        self.countsmat_ = model.countsmat_
        self.n_states_ = model.n_states_
        self.theta0_ = model.theta_

        all_theta, diag, walltime = self.sample()
        self.all_theta_ = all_theta
        self.diag_ = diag
        self.walltime_ = walltime
        return self

    def sample(self):
        assert pyhmc_imported, 'the ``pyhmc`` package is required.'
        alpha = self.prior_alpha
        beta = self.prior_beta
        if np.isscalar(self.prior_alpha):
            # symmetric dirichlet
            alpha = self.prior_alpha * np.ones(self.n_states_)
        if np.isscalar(self.prior_beta):
            beta = self.prior_beta * np.ones(len(self.theta0_) - self.n_states_)

        def func(theta):
            logp, grad = _log_posterior(theta, self.countsmat_,
                alpha=alpha, beta=beta, n=self.n_states_)
            return logp, grad

        epsilon = self.epsilon / len(self.theta0_)
        start = time.time()
        all_theta, diag = hmc(func, x0=self.theta0_, n_samples=self.n_samples,
                            epsilon=epsilon, n_steps=self.n_steps,
                            display=self.verbose, return_diagnostics=True)

        self._is_dirty = True
        return all_theta, diag, time.time() - start

    def summarize(self):
        counts_nz = np.count_nonzero(self.countsmat_)
        cnz = self.countsmat_[np.nonzero(self.countsmat_)]

        correlation_times = integrated_autocorr2(self.all_timescales_[:, :5])

        fmt = lambda x: ', '.join(['{:.2f}'.format(xx) for xx in x])
        fmt_d = lambda x: ', '.join(['{:d}'.format(xx) for xx in x.astype(int)])

        return """Bayesian Markov State Model
---------------------------
Lag time         : {lag_time}
Alpha            : {alpha}
Beta             : {beta}
n_samples        : {n_samples}
n_steps / sample : {n_steps}
epsilon          : {epsilon}

Number of states  : {n_states}
Rejection Rate    : {rej}
Walltime / sample : {walltime:.3} s
Timescales:
    mean : [{mean_ts}]  units
    stdev: [{std_ts}]  units
Timescale correlation times:
    [{correlation_times}]  HMC steps
Approximate number of independent samples:
    [{independent_samples}]
""".format(
        lag_time=self.lag_time, alpha=self.prior_alpha, beta=self.prior_beta,
        n_samples=self.n_samples, n_steps=self.n_steps, epsilon=self.epsilon,
        n_states=self.n_states_, rej=self.diag_['rej'],
        mean_ts=fmt(self.all_timescales_.mean(0)[:5]),
        std_ts=fmt(self.all_timescales_.std(0)[:5]),
        correlation_times=fmt(correlation_times),
        walltime = self.walltime_ / self.n_samples,
        independent_samples=fmt_d(self.n_samples/correlation_times)
        )

    def _get_eigensystem(self):
        if not self._is_dirty:
            return (self._all_eigenvalues,
                    self._all_left_eigenvectors,
                    self._all_right_eigenvectors)

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_states_ - 1

        k = n_timescales + 1

        self._all_eigenvalues = np.empty((len(self.all_theta_), k))
        self._all_left_eigenvectors = np.empty((len(self.all_theta_), self.n_states_, k))
        self._all_right_eigenvectors = np.empty((len(self.all_theta_), self.n_states_, k))

        # for i, transmat in self.all_transmats_:
        for i in range(len(self.all_theta_)):
            u, lv, rv = _solve_ratemat_eigensystem(
                self.all_theta_[i], k, self.n_states_)
            self._all_eigenvalues[i] = u
            self._all_left_eigenvectors[i] = lv
            self._all_right_eigenvectors[i] = rv

        self._is_dirty = False
        return (self._all_eigenvalues,
                self._all_left_eigenvectors,
                self._all_right_eigenvectors)

    @property
    def all_timescales_(self):
        """Implied relaxation timescales each sample in the ensemble

        Returns
        -------
        timescales : array-like, shape = (n_samples, n_timescales,)
            The longest implied relaxation timescales of the each sample in
            the ensemble.
        """

        us, lvs, rvs = self._get_eigensystem()
        # make sure to leave off equilibrium distribution
        timescales = -1 / us[:,1:]
        return timescales

    @property
    def all_eigenvalues_(self):
        """Eigenvalues of the transition matrices.

        Returns
        -------
        eigs : array-like, shape = (n_samples, n_timescales+1)
            The eigenvalues of each rate matrix in the ensemble
        """
        us, lvs, rvs = self._get_eigensystem()
        return us


    @property
    def all_left_eigenvectors_(self):
        r"""Left eigenvectors, :math:`\Phi`, of each rate matrix in the
        ensemble
        """
        us, lvs, rvs = self._get_eigensystem()
        return lvs

    @property
    def all_right_eigenvectors_(self):
        r"""Right eigenvectors, :math:`\Psi`, of each rate matrix in the
        ensemble
        """
        us, lvs, rvs = self._get_eigensystem()
        return rvs

    @property
    def all_populations_(self):
        us, lvs, rvs = self._get_eigensystem()
        return lvs[:, :, 0]


def _log_posterior(theta, counts, alpha, beta, n):
    """Log of the posterior probability and gradient

    Parameters
    ----------
    theta : ndarray, shape=(n_params,)
        The free parameters of the reversible rate matrix
    counts : ndarray, shape=(n, n)
        The count matrix (sufficient statistics for the likielihood)
    alpha : ndarray, shape=(n,)
        Dirichlet concentration parameters
    beta : ndarray, shape=(n_params-n,)
        Scale parameter for the exponential prior on the symmetric rate
        matrix.
    """
    # likelihood + grad
    logp1, grad = loglikelihood(theta, counts)
    # exponential prior on s_{ij}
    logp2 = lexponential(theta[:-n], beta, grad=grad[:-n])
    # dirichlet prior on \pi
    logp3 = ldirichlet_softmax(theta[-n:], alpha=alpha, grad=grad[-n:])
    logp = logp1 + logp2 + logp3
    return logp, grad
