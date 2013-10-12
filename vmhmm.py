#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division
import sys

import numpy as np
from sklearn.hmm import _BaseHMM
import scipy.special
import scipy.optimize
import scipy.interpolate
from scipy.stats.distributions import vonmises


M_2PI = 2*np.pi
DEBUG = False

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class VonMisesHMM(_BaseHMM):
    def _init(self, obs, params='stmk'):
        super(VonMisesHMM, self)._init(obs, params)
        if (hasattr(self, 'n_features')
                and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))
        self.n_features = obs[0].shape[1]

        if 'm' in params:
            self._means_ = np.random.randn(self.n_components, self.n_features)
        if 'k' in params:
            self._kappas_ = np.ones((self.n_components, self.n_features))

    def _get_means(self):
        """Mean parameters for each state."""
        return self._means_

    def _set_means(self, means):
        means = np.asarray(means)
        if (hasattr(self, 'n_features')
                and means.shape != (self.n_components, self.n_features)):
            raise ValueError('means must have shape '
                             '(n_components, n_features)')
        self._means_ = means.copy()
        self.n_features = self._means_.shape[1]

    means_ = property(_get_means, _set_means)

    def _get_kappas(self):
        """Concentration parameter for each state. If kappa is zero, the
        distriution is uniform. If large, it gets very concentrated about
        mean"""
        return self._kappas_

    def _set_kappas(self, kappas):
        kappas = np.asarray(kappas)
        if (hasattr(self, 'n_features')
                and kappas.shape != (self.n_components, self.n_features)):
            raise ValueError('kappas must have shape '
                             '(n_components, n_features)')
        self._kappas_ = kappas.copy()
        self.n_features = self._kappas_.shape[1]

    kappas_ = property(_get_kappas, _set_kappas)

    def _generate_sample_from_state(self, state, random_state=None):
        """Generate random samples from the output distribution of a given state.

        Returns
        -------
        sample : mp.array, shape=[n_features]
        """
        x = vonmises.rvs(self._kappas_[state], self._means_[state])
        wrapped = x - M_2PI * np.floor(x / M_2PI + 0.5)
        return wrapped

    def _compute_log_likelihood(self, obs):
        """Compute the log likelihood of each observation in each state

        Parameters
        ----------
        obs : np.array, shape=[n_samples, n_features]

        Returns
        -------
        logl : np.array, shape=[n_samples, n_states]
        """
        value = np.array([np.sum(vonmises.logpdf(obs, self._kappas_[i], self._means_[i]), axis=1) for i in range(self.n_components)]).T
        return value

    def _initialize_sufficient_statistics(self):
        stats = super(VonMisesHMM, self)._initialize_sufficient_statistics()
        stats['posteriors'] = []
        stats['obs'] = []
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(VonMisesHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        # Unfortunately, I'm not quite sure how to accumulate sufficient
        # statistics here, because we need to know the mean shifted cosine of
        # the data, which requires knowing the mean. You could do two passes,
        # but that is MUCH more work, since you have to redo the forwardbackward
        # So we'll just accumulate the data.
        stats['posteriors'].append(posteriors)
        stats['obs'].append(obs)

    def _do_mstep(self, stats, params):
        super(VonMisesHMM, self)._do_mstep(stats, params)

        posteriors = np.vstack(stats['posteriors'])
        obs = np.vstack(stats['obs'])
        inv_kappas = np.zeros_like(self._kappas_)

        means_ = np.arctan2(np.dot(posteriors.T, np.sin(obs)),
                            np.dot(posteriors.T, np.cos(obs)))

        for i in range(self.n_features):
            for j in range(self.n_components):
                numerator = np.sum(posteriors[:, j] * np.cos(obs[:, i] - means_[j, i]))
                denominator = np.sum(posteriors[:, j])
                inv_kappas[j, i] = numerator / denominator

        self._kappas_ = inverse_mbessel_ratio(inv_kappas)

        self._means_ = means_


class inverse_mbessel_ratio(object):
    """
    Inverse the function given by the ratio modified Bessel function of the
    first kind of order 1 to the modified Bessel function of the first kind
    of order 0.

    y = A(x) = I_1(x) / I_0(x)

    This function computes A^(-1)(y) by way of a precomputed spline
    interpolation
    """
    def __init__(self, n_points=512):
        self._n_points = n_points
        self._is_fit = False
        self._min_x = 1e-5
        self._max_x = 700

    def _fit(self):
        """We want to do the fitting once, but not at import time since it
        slows down the loading of the interpreter"""
        # Fitting takes about 0.5s on a laptop, and wth 512 points and cubic
        # interpolation, gives typical errors around 1e-9
        x = np.logspace(np.log10(self._min_x), np.log10(self._max_x), self._n_points)
        y = self.bessel_ratio(x)
        self._min = np.min(y)
        self._max = np.max(y)

        # Spline fit the log of the inverse function
        self._spline = scipy.interpolate.interp1d(y, np.log(x), kind='cubic')
        (xj, cvals, k) = self._spline._spline
        self._xj = xj
        self._cvals = cvals[:, 0]
        self._k = k

    def __call__(self, y):
        if not self._is_fit:
            self._fit()

        y = np.asarray(y)
        y = np.clip(y, a_min=self._min, a_max=self._max)

        if not np.all(np.logical_and(0 < y, y < 1)):
            raise ValueError('Domain error. y must be in (0, 1)')
        
        # Faster version. Trying to find the real c code so that we can call
        # this in our hot-loop without the overhead.
        x = np.exp(scipy.interpolate._fitpack._bspleval(y, self._xj, self._cvals, self._k, 0))

        if DEBUG:
            x_slow = np.exp(self._spline(y))
            assert np.all(x_slow == x)
            # for debugging, the line below prints the error in the inverse
            # by printing y - A(A^(-1)(y
            print('error', y - self.bessel_ratio(x))

        return x

    @staticmethod
    def bessel_ratio(x):
        numerator = scipy.special.iv(1, x)
        denominator = scipy.special.iv(0, x)
        return numerator / denominator

# Shadown the class with an instance.
inverse_mbessel_ratio = inverse_mbessel_ratio()
print('0.5', inverse_mbessel_ratio(0.5))
