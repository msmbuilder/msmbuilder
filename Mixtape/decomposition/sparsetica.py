# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
from six import PY2
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
    mixtape.tica.tICA

    References
    ----------
    .. [1] Sriperumbudur, Bharath K., David A. Torres, and Gert RG Lanckriet.
       "A majorization-minimization approach to the sparse generalized eigenvalue
       problem." Machine learning 85.1-2 (2011): 3-39.
    .. [2] Mackey, Lester. "Deflation Methods for Sparse PCA." NIPS. Vol. 21. 2008.

    """

    def __init__(self, n_components=None, offset=1, gamma=0.05,
                 rho=0.01, epsilon=1e-6, tolerance=1e-8, maxiter=10000,
                 verbose=False):
        super(SparseTICA, self).__init__(n_components, offset, gamma)
        self.rho = rho
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.verbose = verbose

    def _solve(self):
        if not self._is_dirty:
            return
        if not np.allclose(self.offset_correlation_, self.offset_correlation_.T):
            raise RuntimeError('offset correlation matrix is not symmetric')
        if not np.allclose(self.covariance_, self.covariance_.T):
            raise RuntimeError('correlation matrix is not symmetric')
        if self.rho <= 0:
            s = super(SparseTICA, self) if PY2 else super()
            return s._solve()

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
                          verbose=self.verbose)

            self._eigenvalues_[i] = u
            self._eigenvectors_[:, i] = v
            A = scdeflate(A, v)

        self._is_dirty = False


def scdeflate(A, x):
    """Schur complement matrix deflation

    Eliminate the influence of a psuedo-eigenvector of A using the Schur complement
    deflation technique from [1]::

        A_new = A - \frac{A x x^T A}{x^T A x}

    Parameters
    ----------
    A : np.ndarray, shape=(N, N)
        A matrix
    x : np.ndarray, shape=(N, )
        A vector, ideally one that is "close to" an eigenvector of A

    Returns
    -------
    A_new : np.ndarray, shape=(N, N)
        A new matrix, determined from A by eliminating the influence of x

    References
    .. [1] Mackey, Lester. "Deflation Methods for Sparse PCA." NIPS. Vol. 21. 2008.
    """
    return A - np.outer(np.dot(A, x), np.dot(x, A)) / np.dot(np.dot(x, A), x)


def speigh(A, B, v_init, rho, eps, tol, tau=None, maxiter=10000, verbose=True):
    """Find a sparse approximate generalized eigenpair.

    The generalized eigenvalue equation, :math:`Av = lambda Bv`,
    can be expressed as a variational optimization ::
    :math:`max_{x} x^T A x  s.t. x^T B x = 1`. We can search for sparse
    approximate eigenvectors then by adding a penalty to the optimization.
    This function solves an approximation to::

    max_{x}   x^T A x - \rho ||x||_0

        s.t.      x^T B x <= 1

    Where `||x||_0` is the number of nonzero elements in `x`. Note that
    because of the ||x||_0 term, that problem is NP-hard. Here, we replace
    the ||x||_0 term with

    rho * \sum_i^N \frac{\log(1 + |x_i|/eps)}{1 + 1/eps}

    which converges to ||x||_0 in the limit that eps goes to zero. This
    formulation can then be written as a d.c. (difference of convex) program
    and solved efficiently. The algorithm is due to [1], and is written
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
        Convergence criteria for the eigensolver.

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
            for i in range(maxiter):
                if np.linalg.norm(x[old_x>tol] - old_x[old_x>tol]) < tol:
                    break
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
            for i in range(maxiter):
                if np.linalg.norm(x[old_x>tol] - old_x[old_x>tol]) < tol:
                    break
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
                    result = problem.solve(solver=cp.ECOS)
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
        pprint('Path [3]: tau != 0')
        old_x.fill(np.inf)
        scaledA = (A / tau + np.eye(length))
        for i in range(maxiter):
            if np.linalg.norm(x[old_x>tol] - old_x[old_x>tol]) < tol:
                break
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
    # return x.dot(A).dot(x), x

    # Proposition 1 and the "variational renormalization" described in [1].
    # Use the sparsity pattern in 'x', but ignore the loadings and rerun an
    # unconstrained GEV problem on the submatrices determined by the nonzero
    # entries in our optimized x

    # What cutoff to use for zeroing out entries in 'x'. We could hard-code
    # something, but reusing the `tolerance` parameter seems fine too.
    sparsecutoff = tol

    mask = (np.abs(x) > sparsecutoff)
    grid = np.ix_(mask, mask)
    Ak, Bk = A[grid], B[grid]  # form the submatrices
    gevals, gevecs = scipy.linalg.eigh(Ak, Bk, eigvals=(Ak.shape[0]-2, Ak.shape[0]-1))
    u = gevals[-1]
    v = np.zeros(length)
    v[mask] = gevecs[:, -1]
    return u, v

if __name__ == '__main__':
    X = np.random.randn(1000, 10)
    X[:,0] += np.sin(np.arange(1000) / 100.0)
    X[:,1] += np.cos(np.arange(1000) / 100.0)

    tica = tICA(n_components=2).fit(X)
    print('tica eigenvector\n', tica.components_[0])
    print('tica eigenvalue\n', tica.eigenvalues_[0])
    print('\ntica eigenvector\n', tica.components_[1])
    print('tica eigenvalue\n', tica.eigenvalues_[1])
    print('\n\n')

    sptica = SparseTICA(n_components=2, rho=0.01, tolerance=1e-6, verbose=False)
    sptica.fit(X)
    print('sptica eigenvector\n', sptica.components_[0])
    print('sptica eigenvalue\n', sptica.eigenvalues_[0])
    print('\nsptica eigenvector\n', sptica.components_[1])
    print('sptica eigenvalue\n', sptica.eigenvalues_[1])
