# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
from six import PY2
import numpy as np
import scipy.linalg
from .tica import tICA
from ..utils import experimental
from .speigh import speigh, scdeflate

__all__ = ['SparseTICA']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class SparseTICA(tICA):
    """Sparse Time-structure Independent Component Analysis (tICA)

    Linear dimensionality reduction which finds sparse linear combinations
    of the input features which decorrelate most slowly. These can be
    used for feature selection and/or dimensionality reduction.

    Parameters
    ----------
    n_components : int, None
        Number of components to keep.
    offset : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+offset].
    gamma : nonnegative float, default=0.05
        L2 regularization strength. Positive `gamma` entails incrementing
        the sample covariance matrix by a constant times the identity,
        to ensure that it is positive definite. The exact form of the
        regularized sample covariance matrix is ::

            covariance + (gamma / n_features) * Tr(covariance) * Identity

        where :math:`Tr` is the trace operator.
    rho : positive float
        Controls the sparsity. Higher values of rho gives more
        sparse solutions. rho=0 corresponds to standard tICA
    epsilon : positive float, default=1e-6
        epsilon should be a number very close to zero, which is used to
        construct the approximation to the L_0 penality function. However,
        when it gets *too* close to zero, the solvers may report feasability
        problems due to numberical stability.
    tolerance : positive float
        Convergence critera for the sparse generalized eigensolver.
    maxiter : int
        Maximum number of iterations for the sparse generalized eigensolver.
    verbose : bool
        Print verbose information from the sparse generalized eigensolver.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    offset_correlation_ : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix, `C=E[(x_t)^T x_{t+lag}]`.
    eigenvalues_ : array-like, shape (n_features,)
        Psuedo-eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.
    eigenvectors_ : array-like, shape (n_components, n_features)
        Sparse psuedo-eigenvectors of the tICA generalized eigenproblem. The
        vectors give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium.
    means_ : array, shape (n_features,)
        The mean of the data along each feature
    n_observations_ : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences_ : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.
    timescales_ : array-like, shape (n_components,)
        The implied timescales of the tICA model, given by
        -offset / log(eigenvalues)

    See Also
    --------
    msmbuilder.decomposition.tICA

    References
    ----------
    .. [1] McGibbon, R. T. and V. S. Pande "Identification of sparse, slow
       reaction coordinates from molular dynamics simulations" In preparation.
    .. [1] Sriperumbudur, B. K., D. A. Torres, and G. R. Lanckriet.
       "A majorization-minimization approach to the sparse generalized eigenvalue
       problem." Machine learning 85.1-2 (2011): 3-39.
    .. [3] Mackey, L. "Deflation Methods for Sparse PCA." NIPS. Vol. 21. 2008.
    """

    def __init__(self, n_components=None, lag_time=1, gamma=0.05,
                 rho=0.01, epsilon=1e-6, tolerance=1e-8, maxiter=10000,
                 greedy=True, verbose=False):
        super(SparseTICA, self).__init__(n_components, lag_time=lag_time, gamma=gamma)
        self.rho = rho
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.greedy = greedy
        self.maxiter = maxiter
        self.verbose = verbose

    @experimental('SparseTICA')
    def _solve(self):
        if not self._is_dirty:
            return
        if not np.allclose(self.offset_correlation_, self.offset_correlation_.T):
            raise RuntimeError('offset correlation matrix is not symmetric')
        if not np.allclose(self.covariance_, self.covariance_.T):
            raise RuntimeError('correlation matrix is not symmetric')
        if self.rho <= 0:
            return super(SparseTICA, self)._solve()

        A = self.offset_correlation_
        B = self.covariance_ + (self.gamma / self.n_features) * \
            np.trace(self.covariance_) * np.eye(self.n_features)

        tau = max(0, -np.min(scipy.linalg.eigvalsh(A)))
        gevals, gevecs = scipy.linalg.eigh(A, B)
        ind = np.argsort(gevals)[::-1]
        gevecs, gevals = gevecs[:, ind], gevals[ind]

        self._eigenvalues_ = np.zeros((self.n_components))
        self._eigenvectors_ = np.zeros((self.n_features, self.n_components))

        for i in range(self.n_components):
            u, v = speigh(A, B, gevecs[:, i], rho=self.rho, eps=self.epsilon,
                          tol=self.tolerance, tau=tau, maxiter=self.maxiter,
                          greedy=self.greedy, verbose=self.verbose)

            self._eigenvalues_[i] = u
            self._eigenvectors_[:, i] = v
            A = scdeflate(A, v)

        self._is_dirty = False

    def summarize(self):
        """Some summary information."""

        return """Sparse time-structure based Independent Components Analysis (tICA)
------------------------------------------------------------------
n_components        : {n_components}
gamma               : {gamma}
lag_time            : {lag_time}
weighted_transform  : {weighted_transform}
rho                 : {rho}

Top 5 timescales :
{timescales}

Top 5 eigenvalues :
{eigenvalues}
""".format(n_components=self.n_components, lag_time=self.lag_time, rho=self.rho,
           gamma=self.gamma, weighted_transform=self.weighted_transform,
           timescales=self.timescales_[:5], eigenvalues=self.eigenvalues_[:5])
