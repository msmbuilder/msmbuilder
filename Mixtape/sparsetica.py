# Author: Robert McGibbon
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
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

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.linalg
from mixtape.tica import tICA
# import cvxpy as cp
# (^ this import is now done lazily, inside the function)

__all__ = ['SparseTICA']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class SparseTICA(tICA):
    """Sparse Time-structure Independent Component Analysis (tICA)

    Linear dimensionality reduction which finds sparse linear combinations
    of the input features which decorrelate mose slowly. These can be
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
        sparse solutions.
    epsilon : positive float, default=1e-6
        epsilon should be a number very close to zero, which is used to
        construct the approximation to the L_0 penality function. However,
        when it gets *too* close to zero, the solvers may report feasability
        problems due to numberical stability.
    tolerance : positive float
        Convergence critera for the sparse generalized eigensolver.
    verbose : bool
        Print verbose information from the sparse generalized eigensolver.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    offset_correlation_` : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix, `C=E[(x_t)^T x_{t+lag}]`.
    means_ : array, shape (n_features,)
        The mean of the data along each feature
    n_observations : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.

    See Also
    --------
    tICA

    References
    ----------
    ..[1] Sriperumbudur, Bharath K., David A. Torres, and Gert RG Lanckriet.
    "A majorization-minimization approach to the sparse generalized eigenvalue
    problem." Machine learning 85.1-2 (2011): 3-39.
    """

    def __init__(self, n_components=None, offset=1, gamma=0.05,
                 rho=0.01, epsilon=1e-6, tolerance=1e-8, verbose=False):
        self.rho = rho
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.verbose = verbose
        super(SparseTICA, self).__init__(n_components, offset, gamma)

    def eigenvalues_(self):
        raise NotImplementedError()

    def _solve(self):
        A = self.offset_correlation_
        B = self.covariance_ + (self.gamma / self.n_features) * \
            np.trace(self.covariance_) * np.eye(self.n_features)

        tau = max(0, -np.min(scipy.linalg.eigvalsh(A)))
        gevals, gevecs = scipy.linalg.eigh(A, B)
        ind = np.argsort(gevals)[::-1]
        gevecs, gevals = gevecs[:, ind], gevals[ind]
        v_init = gevecs[:, 0]

        u, v = speigh(A, B, v_init, rho=self.rho, eps=self.epsilon, tol=self.tolerance,
                      tau=tau, verbose=self.verbose)
        print('SparseTICA eigenvector\n', v)
        print('SparseTICA eigenvalue\n', u)
        raise NotImplementedError()


def speigh(A, B, v_init, rho, eps, tol, tau=None, verbose=True):
    """Find sparse approximate generalized eigenpairs.

    The generalized eigevalue equation, :math:`Av = lambda Bv`,
    can be expressed as a variational optimization ::
    :math:`max_{x} x^T A x  s.t. x^T B x = 1`. We can search for sparse
    approximate eigenvectors then by adding a penality to the optimization.
    This function solves an approximation to::

    max_{x}   x^T A x - \rho ||x||_0

        s.t.      x^T B x <= 1

    Where `||x||_0` is the number of nonzero elements in `x`. Note that
    because of the ||x||_0 term, that problem is NP-hard. Here, we replace
    the ||x||_0 term with

    rho * \sum_i^N \frac{\log(1 + |x_i|/eps)}{1 + 1/eps}

    which converges to ||x||_0 in the limit that eps goes to zero. This
    formulation can then be written as a d.c. (difference of convex) program
    and solved efficiently. The algorithim is due to [1], and is written
    down on page 15 of the paper.

    Parameters
    ----------
    A : np.ndarray, shape=(N, N)
        A is symmetric matrix, the left-hand-side of the eigenvalue equation.
    B : np.ndarray, shape=(N, N)
        B is a positive semidefinite matrix, the right-hand-side of the
        eigenvalue equation.
    v_init : np.ndarray, shape=(N,)
        Initial guess for the eigenvector. This should probably be computed by
        running the standard generalized eigensolver first.
    rho : float
        Regularization strength. Larger values for rho will lead to more sparse
        solutions.
    eps : float
        Small number, used in the approximation to the L0. Smaller is better
        (closer to L0), but trickier from a numerical standpoint and can lead
        to the solver complaining when it gets too small.
    tol : float
        Convergence critera for the eigensolver.

    Returns
    -------
    u : float
        The approximate eigenvalue.
    v_final : np.ndarray, shape=(N,)
        The sparse approximate eigenvector

    Notes
    -----
    This function requires the convex optimization library CVXPY [2].

    References
    ----------
    ..[1] Sriperumbudur, Bharath K., David A. Torres, and Gert RG Lanckriet.
    "A majorization-minimization approach to the sparse generalized eigenvalue
    problem." Machine learning 85.1-2 (2011): 3-39.
    ..[2] https://github.com/cvxgrp/cvxpy
    """
    try:
        import cvxpy as cp
    except:
        raise ImportError(
            "Could not import cvxpy, a required package for SparseTICA"
            "See https://github.com/cvxgrp/cvxpy for details")


    pprint = print
    if not verbose:
        pprint = lambda *args : None
    length = A.shape[0]
    x = v_init

    old_x = np.empty(length)
    rho_e = rho / np.log(1 + 1.0/eps)
    b = np.diag(B)
    B_is_diagonal = np.all(np.diag(b) == B)

    if tau == 0:
        if B_is_diagonal:
            pprint('Path [1]: tau=0, diagonal B')
            old_x.fill(np.inf)
            while np.linalg.norm(x-old_x) > tol:
                pprint('x', x)
                old_x = x
                w = 1.0 / (np.abs(x) + eps)
                Ax = A.dot(x)
                absAx = np.abs(Ax)

                if rho_e < 2 * np.max(absAx.dot(w**(-1))):
                    # line 9 of Algorithim 1
                    gamma = absAx - (rho_e/2) * w
                    x_num = np.maximum(gamma, 0) * np.sign(Ax)
                    x_den = b * np.sqrt(np.sum(gamma**2 / b))
                    x = x_num / x_den
                else:
                    x = np.zeros(length)
        else:
            pprint('Path [2]: tau=0, general B')
            old_x.fill(np.inf)
            while np.linalg.norm(x-old_x) > tol:
                pprint('x: ', x)
                old_x = x
                w = 1.0 / (np.abs(x) + eps)
                Ax = A.dot(x)
                absAx = np.abs(Ax)

                if rho_e < 2 * np.max(absAx.dot(w**(-1))):
                    gamma = absAx - (rho_e / 2.0 * w)
                    S = np.diag(np.sign(Ax))
                    SBSInv = scipy.linalg.pinv(S.dot(B).dot(S))

                    # solve for lambda, line 20 of Algorithm 1
                    lambd_ = cp.Variable(length)
                    objective = cp.Minimize(cp.quad_form(gamma + lambd_, SBSInv))
                    constraints = [lambd_ >= 0]
                    problem = cp.Problem(objective, constraints)
                    result = problem.solve()
                    if not problem.status == cp.OPTIMAL:
                        raise ValueError(problem.status)
                    lambd = np.asarray(lambd_.value).flatten()

                    temp = SBSInv.dot(gamma + lambd)
                    x_num = S.dot(temp)
                    x_den = np.sqrt((gamma + lambd).dot(temp))
                    x = x_num / x_den
                else:
                    x = np.zeros(length)

    else:
        print('Path [3]: tau != 0')
        old_x.fill(np.inf)
        scaledA = (A / tau + np.eye(length))
        while np.linalg.norm(x-old_x) > tol:
            pprint('x', x)
            old_x = x
            W = np.diag(1.0 / (np.abs(x) + eps))

            x_ = cp.Variable(length)

            term1 = cp.square(cp.norm((x_ - scaledA.dot(x))))
            term2 = (rho_e / tau) * cp.norm(W * x_, p=1)

            objective = cp.Minimize(term1 + term2)
            constraints = [cp.quad_form(x_, B) <= 1]
            problem = cp.Problem(objective, constraints)
            result = problem.solve(solver=cp.ECOS)

            if not problem.status == cp.OPTIMAL:
                raise ValueError(problem.status)
            x = np.asarray(x_.value).flatten()

    pprint('\nxf:', x)
    return x.dot(A).dot(x), x


if __name__ == '__main__':
    X = np.random.randn(1000, 10)
    X[:,0] = np.sin(np.arange(1000) / 100.0) #+ np.random.randn(1000)*0.1
    #X[:,1] = np.cos(np.arange(1000) / 100.0) + np.random.randn(1000)*0.1

    tica = tICA(n_components=2).fit(X)
    print('tica eigenvector\n', tica.eigenvectors_[0])
    print('tica eigenvalue\n', tica.eigenvalues_[0])


    tica = SparseTICA(n_components=2, rho=0.01, tolerance=1e-6, verbose=False)
    tica.fit(X)
    tica._solve()

