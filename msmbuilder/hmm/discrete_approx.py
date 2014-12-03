"""Discrete approximations to continuous distributions"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.misc
import scipy.linalg
import scipy.optimize
from mdtraj.utils import ensure_type

__all__ = ['discrete_approx_mvn', 'NotSatisfiableError']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class NotSatisfiableError(Exception):
    pass


def discrete_approx_mvn(X, means, covars, match_variances=True):
    """Find a discrete approximation to a multivariate normal distribution.

    The method employs find the discrete distribution with support only at the
    supplied points X with minimal K-L divergence to a target multivariate
    normal distribution under the constraints that the mean and variance
    of the discrete distribution match the normal distribution exactly.

    Parameters
    ----------
    X : np.ndarray, shape=(n_points, n_features)
        The allowable points
    means : np.ndarray, shape=(n_features)
        The mean vector of the MVN
    covars : np.ndarray, shape=(n_features, n_features) or shape=(n_features,)
        If covars is 2D, it's interpreted as the covariance matrix for
        the model. If 1D, we assume a diagonal covariance matrix with the
        specified diagonal entries.
    match_variances : bool, optimal
        When True, both the means and the variances of the discrete distribution
        are constrained. Under some circumstances, this is not satisfiable (e.g.
        if there aren't enough samples

    Returns
    -------
    weights : np.ndarray, shape=(n_samples,)
        The weight for each of the points in X in the resulting
        discrete probability distribution

    Notes
    -----
    The discrete distribution is one that has mass only at the specified
    points. It can therefore be parameterized by a set of weights on each
    point. If :math:`\{X_i\}` is the set of allowable points, and
    :math:`\{w_i\}` are the weights, then our discrete distribution has
    the form

    .. math::

        p(y; w) = w_i \sum \delta(y - X_i).

    We chose the :math:`w_i` by minimizing the K-L divergence from the our
    discrete distribution to the desired multivariate normal subject to a
    constraint that the first moments of the discrete distribution match
    the mean of the multivariate normal exactly, and that the variances
    also match. Let :math:`q(x)` be the target distribution. The optimal
    weights are then

    .. math::

        min_{\{w_i\}} \sum_i p(X_i; w) \log \frac{p(X_i; w)}{q(X_i)}

    subject to

    .. math::

        \sum_i (X_i)       p(X_i; w) = \int_\Omega (x)    q(x) = \mu,
        \sum_i (X_i-mu)**2 p(X_i; w) = \int_\Omega (x-mu) q(x).

    References
    ----------
    .. [1] Tanaka, Ken'ichiro, and Alexis Akira Toda. "Discrete approximations
    of continuous distributions by maximum entropy." Economics Letters 118.3
    (2013): 445-450.
    """
    X = ensure_type(np.asarray(X), dtype=np.float32, ndim=2, name='X', warn_on_cast=False)
    means = ensure_type(np.asarray(means), np.float64, ndim=1, name='means', warn_on_cast=False)
    covars = np.asarray(covars)

    # Get the un-normalized probability of each point X_i in the MVN
    # `prob` are the q(X_i) in the mathematics
    # `moments` are the \bar{T} that we want to match.
    if covars.ndim == 1:
        # diagonal covariance case
        if not len(covars) == len(means):
            raise ValueError('Shape Error: covars and means musth have the same length')
        prob = np.exp(-0.5 * np.sum(1. / np.sqrt(covars) * (X - means) ** 2, axis=1))
        moments = np.concatenate((means, covars)) if match_variances else means

    elif covars.ndim == 2:
        if not (covars.shape[0] == len(means) and covars.shape[1] == len(means)):
            raise ValueError('Shape Error: covars must be square, with size = len(means)')

        # full 2d covariance matrix
        cv_chol = scipy.linalg.cholesky(covars, lower=True)
        cv_sol = scipy.linalg.solve_triangular(cv_chol, (X - means).T, lower=True).T
        prob = np.exp(-0.5 * (np.sum(cv_sol ** 2, axis=1)))
        moments = np.concatenate((means, np.diag(covars))) if match_variances else means
    else:
        raise ValueError('covars must be 1D or 2D')

    # this is T(x_i) for each X_i
    moment_contributions = np.hstack((X, (X - means) ** 2)) if match_variances else X

    def objective_and_grad(l):
        dot = np.dot(moment_contributions, l)
        lse = scipy.misc.logsumexp(dot, b=prob)
        # value of the objective function
        obj_value = lse - np.dot(l, moments)

        # gradient of objective function
        dot_max = dot.max(axis=0)

        exp_term = np.sum(moment_contributions * (prob * np.exp(dot - dot_max)).reshape(-1, 1), axis=0)
        log_numerator = np.log(exp_term) + dot_max
        grad_value = np.exp(log_numerator - lse) - moments

        return obj_value, grad_value

    result = scipy.optimize.minimize(
        objective_and_grad, jac=True, x0=np.ones_like(moments), method='BFGS')
    if not result['success']:
        raise NotSatisfiableError()

    dot = np.dot(moment_contributions, result['x'])
    log_denominator = scipy.misc.logsumexp(dot, b=prob)
    weights = prob * np.exp(dot - log_denominator)
    if not np.all(np.isfinite(weights)):
        raise NotSatisfiableError()
    weights = weights / np.sum(weights)
    return weights


if __name__ == '__main__':
    np.random.seed(10)
    import matplotlib.pyplot as pp
    length = 100

    X = np.random.uniform(low=-5, high=5, size=(length, 1))
    weights = discrete_approx_mvn(X, [0], [2])
    pp.title('dot(weights, X) = %.5f, dot(weights, X**2)=%f' %
             (np.dot(weights, X), np.dot(weights, X ** 2)))
    for i in range(length):
        pp.plot([X[i, 0], X[i, 0]], [0, weights[i]])

    pp.figure()
    X = np.random.uniform(low=-2, high=2, size=(length, 1))
    weights = discrete_approx_mvn(X, [0], [1])
    pp.title('dot(weights, X) = %.5f, dot(weights, X**2)=%f' %
             (np.dot(weights, X), np.dot(weights, X ** 2)))
    for i in range(length):
        pp.plot([X[i, 0], X[i, 0]], [0, weights[i]])

    pp.show()
