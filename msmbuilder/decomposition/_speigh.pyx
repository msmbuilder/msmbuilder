from __future__ import print_function

from libc.math cimport sqrt, fabs, log

import numpy as np
import scipy.linalg
import scipy.special

include "cy_blas.pyx"

cdef double SPEIGH_TAU_MIN = 0.1
cdef double ADMM_RHO_INITIAL = 16
cdef double ADMM_PENALTY_MU_FACTOR = 10
cdef double ADMM_PENALTY_TAU_FACTOR = 2
cdef double ADMM_PENALTY_RHO_MAX = 2**10
cdef double ADMM_PENALTY_RHO_MIN = 2**-10


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
    ----------
    ... [1] Mackey, Lester. "Deflation Methods for Sparse PCA." NIPS. Vol.
        21. 2008.
    """
    return A - np.outer(np.dot(A, x), np.dot(x, A)) / np.dot(np.dot(x, A), x)


def speigh(double[:, ::1] A, double[:, ::1] B, double rho, double eps=1e-6,
           double tol=1e-8, int maxiter=100, int verbose=False, int method=1):
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

    which converges to ||x||_0 in the limit that eps goes to zero.

    Parameters
    ----------
    A : np.ndarray, shape=(N, N)
        A is symmetric matrix, the left-hand-side of the eigenvalue equation.
    B : np.ndarray, shape=(N, N)
        B is a positive semidefinite matrix, the right-hand-side of the
        eigenvalue equation.
    rho : float
        Regularization strength. Larger values for rho will lead to more sparse
        solutions.
    eps : float
        Small number, used in the approximation to the L0. Smaller is better
        (closer to L0), but trickier from a numerical standpoint and can lead
        to the solver complaining when it gets too small.
    tol : float
        Convergence criteria for the eigensolver.
    maxiter : int
        Maximum number of iterations.
    method : int
        1: Default ADMM solver.
        2. CVXOPT interior point solver for the QCQP.

    Returns
    -------
    u : float
        The approximate eigenvalue.
    x : np.ndarray, shape=(N,)
        The sparse approximate eigenvector

    References
    ----------
    ... [1] Sriperumbudur, B. K., D. A. Torres, and G. R. G. Lanckriet.
        "A majorization-minimization approach to the sparse generalized
        eigenvalue problem." Machine learning 85.1-2 (2011): 3-39.
    ... [2] McGibbon, R. T. and V. S. Pande, in preparation (2015)
    """
    cdef int N = len(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix')
    if B.shape[0] != B.shape[1]:
        raise ValueError('B must be square matrix')
    if A.shape[0] != B.shape[0]:
        raise ValueError('Wrong B dimensions (%d, %d) should be (%d, %d)' % (
            B.shape[0], B.shape[1], A.shape[0], A.shape[1]))

    cdef int i              # iteration counter
    cdef int j
    cdef double f, old_f    # current and old objective function
    cdef double[::1] x      # solution vector to be iterated
    cdef double rho_e = rho / scipy.special.log1p(1/eps)
    cdef double tau = SPEIGH_TAU_MIN + max(0, -np.min(scipy.linalg.eigvalsh(A)))
    f, old_f = np.inf, np.inf

    cdef double[::1] Ax = np.empty(N)        # Matrix vector product: dot(A, x)
    cdef double[::1] w = np.empty(N)
    cdef double[::1] b = np.empty(N)
    # Initialize solver from dominant generalized eigenvector (unregularized
    # solution)
    x = scipy.linalg.eigh(A, B, eigvals=(N-1, N-1))[1][:,0]

    cdef double[::1] B_eigvals
    cdef double[:, ::1] B_eigvecs
    if method == 1:
        B_eigvals, B_eigvecs = map(np.ascontiguousarray, scipy.linalg.eigh(B))

    for i in range(maxiter):
        old_f = f
        cdgemv_N(A, x, Ax)   # Ax = dot(A, x)

        cddot(Ax, x, &f)     # f = Ax.dot(x) - rho_e*sum_i(log(|x_i| + eps))
        for j in range(N):
            f -= rho_e * log(fabs(x[j]) + eps)

        if verbose:
            print("QCQP objective=%.5f" % f)
        if abs(old_f - f) < tol:
            break

        # b = np.dot(A, x)/tau + x
        for j in range(N):
            b[j] = Ax[j]/tau + x[j]

        # w = rho_e / (2 * tau * (|x| + eps))
        for j in range(N):
            w[j] = rho_e / (2*tau*(fabs(x[j]) + eps))

        if method == 1:
            f2 = solve_admm(b, w, B_eigvals, B_eigvecs, x, tol=tol, maxiter=maxiter, verbose=verbose)
        elif method == 2:
            x, f2 = solve_cvxpy(b, w, B)
        else:
            raise ValueError('Unknown method')

        if verbose:
            print("ADMM objective=%.5f" % f2)

    if verbose:
        print('Optimized vector (before renormalization)')
        print(np.asarray(x))

    # Proposition 1 and the "variational renormalization" described in [1].
    # Use the sparsity pattern in 'x', but ignore the loadings and rerun an
    # unconstrained GEV problem on the submatrices determined by the nonzero
    # entries in our optimized x
    mask = (np.abs(x) > 0)
    grid = np.ix_(mask, mask)
    Ak, Bk = np.asarray(A)[grid], np.asarray(B)[grid]  # form the submatrices

    if len(Ak) == 0:
        u, v = 0, np.zeros(N)
    elif len(Ak) == 1:
        v = np.zeros(N)
        v[mask] = 1.0 / np.sqrt(Bk[0,0])
        u = Ak[0,0] / Bk[0,0]
    else:
        gevals, gevecs = scipy.linalg.eigh(
            Ak, Bk, eigvals=(Ak.shape[0]-1, Ak.shape[0]-1))
        # Usually slower to use sparse linear algebra here
        # gevals, gevecs = scipy.sparse.linalg.eigsh(
        #     A=Ak, M=Bk, k=1, v0=x[mask], which='LA')
        u = gevals[0]
        v = np.zeros(N)
        v[mask] = gevecs[:, 0]
        v *= np.sign(np.sum(v))
        v[v==-0] = 0

    return u, v


def solve_cvxpy(b, w, B):
    """Solve a convex optimization problem using cvxpy

    Minimize    1/2 ||x-b||^2 + ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B : array, shape=(n,n)

    Returns
    -------
    xf : array, shape=(n,)
        The optimal value of x
    fun : float
        The value of the objective function at ``xf``
    """
    import cvxpy as cp
    n = len(b)
    x = cp.Variable(n)
    #objective = cp.Minimize(
    #    0.5 * cp.norm2(x-b)**2 + lambd*cp.norm1(cp.diag(w) * x))

    # more accurate this way, expanding out the norm
    objective = cp.Minimize(
        0.5*cp.norm2(x)**2 - x.T*np.asarray(b) + cp.norm1(cp.diag(np.asarray(w)) * x))

    constraints = [cp.quad_form(x, np.asarray(B)) <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != 'optimal':
        print(problem, problem.status)
        raise ValueError('No solution found')

    x = np.asarray(x.value)[:,0]
    x[np.abs(x) < 1e-6] = 0
    fun = 0.5*np.dot(x-b, x-b) + np.sum(np.abs(np.multiply(w, x)))
    return x, fun


cpdef double solve_admm(const double[::1] b, const double[::1] w,
                        const double[::1] B_eigvals,
                        const double[:, ::1] B_eigvecs, double[::1] x,
                        double tol=1e-6, int maxiter=100, int verbose=0):
    """Solve a particular convex optimization problem with ADMM

    Minimize    1/2 ||x-b||^2 + ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B_eigvals : array, shape=(n,)
    B_eigvecs : array, shape=(n,n)
    x : array, shape=(n,)
    tol : float, default=1e-6
    maxiter : int, default=100
    """

    cdef int i, j
    cdef int N = len(b)

    if not (len(b) == len(w) == len(x) == len(B_eigvals) == B_eigvecs.shape[0] == B_eigvecs.shape[1]):
        raise ValueError('Incompatible matrix dimensions')

    cdef double rho = ADMM_RHO_INITIAL
    cdef double[::1] z = np.copy(x)
    cdef double[::1] z_old = np.copy(x)
    cdef double[::1] u = np.zeros(N)

    # temps
    cdef double r_norm2, s_norm2
    cdef double[::1] absw = np.abs(w)
    cdef double[::1] v = np.empty(N)
    cdef double[::1] b_rho_zu = np.empty(N)
    cdef double[::1] r = np.empty(N)
    cdef double[::1] s = np.empty(N)

    for i in range(maxiter):
        z_old[:] = z[:]

        for j in range(N):
            b_rho_zu[j] = b[j] + rho*(z[j] - u[j])

        soft_thresh(absw, b_rho_zu, x)
        for j in range(N):
            x[j] = x[j] / (rho+1.0)

        for j in range(N):
            v[j] = x[j] + u[j]

        project(v, B_eigvals, B_eigvecs, z)

        for j in range(N):
            r[j] = x[j] - z[j]               # primal residual
            s[j] = rho * (z[j] - z_old[j])   # dual residual
            u[j] = u[j] + r[j]

        # primal residual: r_norm2 = ||r||^2
        cddot(r, r, &r_norm2)
        cddot(s, s, &s_norm2)

        if verbose > 1:
            print(' rho', rho, ' residuals ', sqrt(r_norm2), sqrt(s_norm2))

        if r_norm2 < N*tol*tol and s_norm2 < N*tol*tol:
            break

        # Varying the penalty parameter, eq. (3.13)
        if r_norm2 > ADMM_PENALTY_MU_FACTOR**2 * s_norm2 and rho < ADMM_PENALTY_RHO_MAX:
            rho *= ADMM_PENALTY_TAU_FACTOR
            for j in range(N):
                u[j] /= ADMM_PENALTY_TAU_FACTOR
        if s_norm2 > ADMM_PENALTY_MU_FACTOR**2 * r_norm2 and rho > ADMM_PENALTY_RHO_MIN:
            rho /= ADMM_PENALTY_TAU_FACTOR
            for j in range(N):
                u[j] *= ADMM_PENALTY_TAU_FACTOR

    return 0.5*np.dot(x,x) - np.dot(x,b) + 0.5*np.dot(b,b) + np.sum(np.abs(np.multiply(w, x)))


###############################################################################
## Utilities
###############################################################################

cdef soft_thresh(const double[::1] k, const double[::1] a, double[::1] out):
    cdef int i
    cdef int N = len(k)
    if len(a) != len(k):
        raise ValueError('Incompatible matrix dimensions')

    for i in range(N):
        if a[i] > k[i]:
            out[i] = a[i] - k[i]
        elif a[i] < -k[i]:
            out[i] = a[i] + k[i]
        else:
            out[i] = 0



cpdef project(const double[::1] a, const double[::1] w, const double[:, ::1] V,
              double[::1] out):
    """ Compute the projection of a vector a onto an ellipsoid

        Minimize_x ||x-a||^2
        subject to x^TBx <= 1

    where (w, V) are the eigenvalues and eigenvectors of the positive definite
    matrix B.

    Parameters
    ----------
    a : array, shape=(n,)
        The vector to project
    w : array, shape=(n,)
        eigenvalues of B
    V : array, shape=(n,n)
        eigenvectors of B
    out : array, shape=(n,)
        On exit, the results will be written here
    """
    cdef int i, j
    cdef int N = len(a)
    cdef double mu, c2w_sum, G, Gp, muw1, delta
    cdef double[::1] c = np.empty(N)
    cdef double[::1] c2w = np.empty(N)
    if not (len(a) == len(w) == V.shape[0] == V.shape[1]):
        raise ValueError('Incompatible matrix dimensions')

    cdgemv_T(V, a, c)   # c = V.T.dot(a)

    c2w_sum = 0
    for j in range(N):
        c2w[j] = c[j]*c[j] * w[j]
        c2w_sum += c2w[j]

    if c2w_sum < 1:
        out[:] = a[:]
        return

    mu = 0
    for i in range(100):
        G = -1
        Gp = 0
        for j in range(N):
            muw1 = (mu*w[j]+1)
            G += c2w[j] / (muw1*muw1)
            Gp -= 2*c2w[j]*w[j] / (muw1*muw1*muw1)
        delta = -G/Gp
        mu = mu + delta
        if fabs(delta) < 1e-12:
            break

    for j in range(N):
        c2w[j] = c[j] / (mu*w[j] + 1)
    cdgemv_N(V, c2w, out)
    # return out
