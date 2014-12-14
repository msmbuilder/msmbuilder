# cython: boundscheck=False, cdivision=True, wraparound=False, c_string_encoding=ascii
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""Implementation of the log-likelihood function and gradient for a
continuous-time reversible Markov model sampled at a regular interval.

For details on the parameterization of the `\theta` vector, refer to
the documentation in docs/ratematrix.rst

Performance
-----------
This is pretty low-level cython code. Everything is explicitly typed, and
we use cython memoryviews. Using the wrappers from `cy_blas.pyx`, we make
direct calls to the FORTRAN BLAS matrix multiply using the function pointers
that scipy exposes.

There are assert statements which provide some clarity, but are slow.
They are compiled away by default, unless `python setup.py install` is
run with the flag `--debug` on the command line.
"""

from __future__ import print_function
import numpy as np
from numpy import (zeros, allclose, real, ascontiguousarray, asfortranarray)
import scipy.linalg
from numpy cimport npy_intp
from libc.math cimport sqrt, log, exp
from libc.string cimport memset, strcmp

include "cy_blas.pyx"
include "config.pxi"
include "triu_utils.pyx"      # ij_to_k() and k_to_ij()
include "binary_search.pyx"   # bsearch()
include "_ratematrix_support.pyx"


cpdef int build_ratemat(const double[::1] exptheta, npy_intp n, const npy_intp[::1] inds,
                        double[:, ::1] out, char* which=b'K'):
    """Build the reversible rate matrix K or symmetric rate matrix, S,
    from the free parameters, `\theta`

    Parameters
    ----------
    exptheta : array
        The element-wise exponential of the free parameters, `\theta`.
        These values are the linearized elements of the upper triangular portion
        of the symmetric rate matrix, S, followed by the equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    inds : array, optional (default=None)
        Sparse linearized triu indices exptheta. If not supplied, exptheta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(exptheta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `exptheta` correspond.
    which : {'S', 'K'}
        Whether to build the matrix S or the matrix K
    out : [output], array shape=(n, n)
        On exit, out contains the matrix K or S

    Notes
    -----
    The last `n` elements of exptheta must be nonzero, since they parameterize
    the equilibrium populations, so even with the sparse parameterization,
    `len(u)` must be greater than or equal to `n`.

    With the sparse parameterization, the following invariant holds. If

        inds = indices_of_nonzero_elements(exptheta)

    Then,

        build_ratemat(exptheta, n, None) == build_ratemat(exptheta[inds], n, inds)
    """
    cdef npy_intp u = 0, k = 0, i = 0, j = 0, n_triu = 0
    cdef double s_ij, K_ij, K_ji
    cdef double[::1] pi
    if DEBUG:
        assert out.shape[0] == n
        assert out.shape[1] == n
        assert inds is None or inds.shape[0] >= n
        assert ((exptheta.shape[0] == inds.shape[0]) or
                (inds is None and exptheta.shape[0] == n*(n-1)/2 + n))
        assert np.all(np.asarray(out) == 0)
    if inds is None:
        n_triu = n*(n-1)/2
    else:
        n_triu = inds.shape[0] - n

    pi = exptheta[n_triu:]

    for k in range(n_triu):
        if inds is None:
            u = k
        else:
            u = inds[k]

        k_to_ij(u, n, &i, &j)
        s_ij = exptheta[k]

        if DEBUG:
            assert 0 <= u < n*(n-1)/2

        K_ij = s_ij * sqrt(pi[j] / pi[i])
        K_ji = s_ij * sqrt(pi[i] / pi[j])
        if strcmp(which, 'S') == 0:
           out[i, j] = s_ij
           out[j, i] = s_ij
        else:
            out[i, j] = K_ij
            out[j, i] = K_ji
        out[i, i] -= K_ij
        out[j, j] -= K_ji

    if DEBUG:
        assert np.allclose(np.array(out).sum(axis=1), 0.0)
        assert np.allclose(scipy.linalg.expm(np.array(out)).sum(axis=1), 1)
        assert np.all(0 < scipy.linalg.expm(np.array(out)))
        assert np.all(1 > scipy.linalg.expm(np.array(out)))
    return 0


cpdef double dK_dtheta_A(const double[::1] exptheta, npy_intp n, npy_intp u,
                         const npy_intp[::1] inds, const double[:, ::1] A,
                         double[:, ::1] out=None) nogil:
    """Compute the sum of the Hadamard (element-wise) product of the
    derivative of (the rate matrix, `K`, with respect to the free
    parameters,`\theta`, dK_ij / dtheta_u) and another matrix, A.

    Since dK/dtheta_u is a sparse matrix with a known sparsity structure, it's
    more efficient to just do sum as we construct it, and never
    save the matrix elements directly.

    Parameters
    ----------
    exptheta : array
        The element-wise exponential of the free parameters, `\theta`.
        These values are the linearized elements of the upper triangular portion
        of the symmetric rate matrix, S, followed by the equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    u : int
        The index, `0 <= u < len(exptheta)` of the element in `theta` to
        construct the derivative of the rate matrix, `K` with respect to.
    inds : array, optional (default=None)
        Sparse linearized triu indices exptheta. If not supplied, exptheta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(exptheta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `exptheta` correspond.
    A : array of shape=(n, n), optional
        If not None, an arbitrary (n, n) matrix to be multiplied element-wise
        with the derivative of the rate matrix, dKu.
    out : [output], optional array of shape=(n, n)
        If not None, out will contain the matrix dKu on exit.

    Returns
    -------
    s : double
        The sum of the element-wise product of dK/du and A, if A is not None.
    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    cdef npy_intp a, i, j, n_triu, uu, kk
    cdef double dK_i, s_ij, dK_ij, dK_ji
    cdef double sum_elem_product = 0
    cdef double[::1] pi
    if DEBUG:
        assert out.shape[0] == n and out.shape[1] == n
        assert A.shape[0] == n and A.shape[1] == n
        assert inds is None or inds.shape[0] >= n
        assert ((exptheta.shape[0] == inds.shape[0]) or
                (inds is None and exptheta.shape[0] == n_S_triu + n))
    if inds is None:
        n_triu = n*(n-1)/2
    else:
        n_triu = inds.shape[0] - n
    if out is not None:
        memset(&out[0,0], 0, n*n*sizeof(double))

    pi = exptheta[n_triu:]
    uu = u
    if inds is not None:
        # if inds is None, then `u` indexes right into the linearized
        # upper triangular rate matrix. Othewise, it's uu=inds[u] that indexes
        # into the upper triangular rate matrix.
        uu = inds[u]

    if uu < n_S_triu:
        # the perturbation is to the triu rate matrix
        # first, use the linear index, u, to get the (i,j)
        # indices of the symmetric rate matrix
        k_to_ij(uu, n, &i, &j)

        s_ij = exptheta[u]
        dK_ij = s_ij * sqrt(pi[j] / pi[i])
        dK_ji = s_ij * sqrt(pi[i] / pi[j])

        if A is not None:
            sum_elem_product = (
                A[i,j]*dK_ij + A[j,i]*dK_ji
              - A[i,i]*dK_ij - A[j,j]*dK_ji
            )

        if out is not None:
            out[i, j] = dK_ij
            out[j, i] = dK_ji
            out[i, i] -= dK_ij
            out[j, j] -= dK_ji

    else:
        # the perturbation is to the equilibrium distribution

        # `i` is now the index, in `pi`, of the perturbed element
        # of the equilibrium distribution.
        i = u - n_triu

        # the matrix dKu has 1 nonzero row, 1 column, and the diagonal. e.g:
        #
        #    x     x
        #      x   x
        #        x x
        #    x x x x x x
        #          x x
        #          x   x

        for j in range(n):
            if j == i:
                continue

            k = ij_to_k(i, j, n)
            kk = k
            if inds is not None:
                kk = bsearch(inds, k)
            if kk == -1:
                continue

            s_ij = exptheta[kk]
            dK_ij = -0.5 * s_ij * sqrt(pi[j] / pi[i])
            dK_ji = 0.5  * s_ij * sqrt(pi[i] / pi[j])

            if A is not None:
                sum_elem_product += (
                    A[i,j]*dK_ij + A[j,i]*dK_ji
                  - A[i,i]*dK_ij - A[j,j]*dK_ji
                )

            if out is not None:
                out[i, j] = dK_ij
                out[j, i] = dK_ji
                out[i, i] -= dK_ij
                out[j, j] -= dK_ji

    return sum_elem_product


def loglikelihood(const double[::1] theta, const double[:, ::1] counts, npy_intp n,
                  const npy_intp[::1] inds=None, double t=1):
    """Log likelihood and gradient of the log likelihood of a continuous-time
    Markov model.

    Parameters
    ----------
    theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
        The free parameters of the model. These values are the (possibly sparse)
        linearized elements of the log of the  upper triangular portion of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    counts : array of shape = (n, n)
        The matrix of observed transition counts.
    n : int
        The size of `counts`
    inds : array, optional (default=None)
        Sparse linearized triu indices theta. If not supplied, theta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(theta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `theta` correspond.
    t : double
        The lag time.

    Returns
    -------
    logl : double
        The log likelihood of the parameters.
    grad : array of shape = (n*(n-1)/2 + n)
        The gradient of the log-likelihood with respect to `\theta`
    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    if not (counts.shape[0] == n and counts.shape[1] == n):
        raise ValueError('counts must be n x n')
    if not (inds is None or inds.shape[0] >= n):
        raise ValueError('inds must be None (dense) or an array longer than n')
    if inds is not None:
        if not np.all(inds == np.unique(inds)):
            raise ValueError('inds must be sorted, without redundant')
    if not ((theta.shape[0] == inds.shape[0]) or
            (inds is None and theta.shape[0] == n_S_triu + n)):
        raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
    if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
        raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')

    cdef npy_intp u, i, j
    cdef npy_intp size = theta.shape[0]
    cdef double logl = 0
    cdef double[::1] grad, exptheta, w
    cdef double[:, ::1] K, T, dT, U, V

    grad = zeros(size)
    exptheta = zeros(size)
    S = zeros((n, n))
    T = zeros((n, n))
    dT = zeros((n, n))

    for u in range(size):
        exptheta[u] = exp(theta[u])

    build_ratemat(exptheta, n, inds, S, 'S')
    if not np.all(np.isfinite(S)):
        # these parameters don't seem good...
        # tell the optimizer to stear clear!
        return -np.inf, ascontiguousarray(grad)

    w, U, V = eigK(S, n, exptheta[size-n:], 'S')
    dT_dtheta(w, U, V, counts, n, t, T, dT)

    with nogil:
        for u in range(size):
            grad[u] = dK_dtheta_A(exptheta, n, u, inds, dT)

        for i in range(n):
            for j in range(n):
                if counts[i, j] > 0:
                    logl += counts[i, j] * log(T[i, j])

    return logl, ascontiguousarray(grad)


def hessian(double[::1] theta, double[:, ::1] counts, npy_intp n, double t=1,
            npy_intp[::1] inds=None):
    """Estimate of the hessian of the log-likelihood with respect to \theta.

    Parameters
    ----------
    theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
        The free parameters of the model. These values are the (possibly sparse)
        linearized elements of the log of the  upper triangular portion of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    counts : array of shape = (n, n)
        The matrix of observed transition counts.
    n : int
        The size of `counts`
    t : double
        The lag time.
    inds : array, optional (default=None)
        Sparse linearized triu indices theta. If not supplied, theta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(theta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `theta` correspond.

    Notes
    -----
    This computation follows equation 3.6 of [1].

    References
    ----------
    ..[1] Kalbfleisch, J. D., and Jerald F. Lawless. "The analysis of panel data
          under a Markov assumption." J. Am. Stat. Assoc. 80.392 (1985): 863-871.

    Returns
    -------
    m : array, shape=(len(theta), len(theta))
        An estimate of the hessian of the log-likelihood
    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    if not (counts.shape[0] == n and counts.shape[1] == n):
        raise ValueError('counts must be n x n')
    if not (inds is None or inds.shape[0] >= n):
        raise ValueError('inds must be None (dense) or an array longer than n')
    if inds is not None:
        if not np.all(inds == np.unique(inds)):
            raise ValueError('inds must be sorted, without redundant')
    if not ((theta.shape[0] == inds.shape[0]) or
            (inds is None and theta.shape[0] == n_S_triu + n)):
        raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
    if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
        raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')

    cdef npy_intp size = theta.shape[0]
    cdef npy_intp u, v, i, j
    cdef double hessian_uv
    cdef double[::1] grad, exptheta, expwt
    cdef double[:, ::1] K, T, Q, dKu,  Au, temp1, temp2

    hessian = zeros((size, size))
    exptheta = zeros(size)
    expwt = zeros(n)
    S = zeros((n, n))
    T = zeros((n, n))
    Q = zeros((n, n))
    dKu = zeros((n, n))
    Au = zeros((n, n))
    temp1 = zeros((n, n))
    temp2 = zeros((n, n))
    rowsums = np.sum(counts, axis=1)

    for u in range(size):
        exptheta[u] = exp(theta[u])

    build_ratemat(exptheta, n, inds, S, 'S')
    w, U, V = eigK(S, n, exptheta[size-n:], 'S')

    for i in range(n):
        expwt[i] = exp(w[i]*t)

    transmat(expwt, U, V, n, temp1, T)  # write transmat into T

    for i in range(n):
        for j in range(n):
            Q[i,j] = -rowsums[i] / T[i, j]

    for u in range(size):
        dK_dtheta_A(exptheta, n, u, inds, None, dKu)
        # Gu = U.T * dKu * V
        cdgemm_TN(U, dKu, temp1)
        cdgemm_NN(temp1, V, temp2)

        # dPu = V (Gu \circ X) U.T
        hadamard_X(w, expwt, t, n, temp2)
        cdgemm_NN(V, temp2, temp1)
        cdgemm_NT(temp1, U, temp2)

        # Bu = V^T (dPu \circ Q) U
        hadamard_inplace(temp2, Q)
        cdgemm_TN(V, temp2, temp1)
        cdgemm_NN(temp1, U, temp2)

        # Au = U (Bu \circ X) V.T
        hadamard_X(w, expwt, t, n, temp2)
        cdgemm_NN(U, temp2, temp1)
        cdgemm_NT(temp1, V, Au)

        for v in range(u, size):
            hessian_uv = dK_dtheta_A(exptheta, n, v, inds, Au)
            hessian[u, v] = hessian_uv
            hessian[v, u] = hessian_uv

    return np.asarray(hessian)


def sigma_K(const double[:, :] covar_theta, const double[::1] theta,
                  npy_intp n, npy_intp[::1] inds=None):
    """Estimate the asymptotic standard deviation (uncertainty in the rate
    matrix, `K`

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
        The free parameters of the model. These values are the (possibly sparse)
        linearized elements of the log of the  upper triangular portion of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    n : int
        The size of `counts`
    inds : array, optional (default=None)
        Sparse linearized triu indices theta. If not supplied, theta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(theta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `theta` correspond.

    Returns
    -------
    sigma_K : array, shape=(n, n)
        Estimate of the element-wise asymptotic standard deviation of the rate matrix, K.
    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    cdef npy_intp u, v, i, j
    cdef double[::1] exptheta
    cdef double[:, ::1] var_K, dKu, dKv, K, eye
    cdef npy_intp size = theta.shape[0]
    if not (inds is None or inds.shape[0] >= n):
        raise ValueError('inds must be None (dense) or an array longer than n')
    if inds is not None:
        if not np.all(inds == np.unique(inds)):
            raise ValueError('inds must be sorted, without redundant')
    if not ((theta.shape[0] == inds.shape[0]) or
            (inds is None and theta.shape[0] == n_S_triu + n)):
        raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
    if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
        raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('counts must be `size` x `size`')

    var_K = zeros((n, n))
    dKu = zeros((n, n))
    dKv = zeros((n, n))
    K = zeros((n, n))
    exptheta = np.exp(theta)
    eye = np.eye(n)

    for u in range(size):
        dK_dtheta_A(exptheta, n, u, inds, None, dKu)
        for v in range(size):
            dK_dtheta_A(exptheta, n, v, inds, None, dKv)
            # this could be optimized, since dKu and dKv are sparse and we
            # know their pattern
            for i in range(n):
                for j in range(n):
                    var_K[i,j] += covar_theta[u,v] * dKu[i,j] * dKv[i,j]

    return np.asarray(np.sqrt(var_K))


def sigma_pi(const double[:, :] covar_theta, const double[::1] theta,
             npy_intp n, npy_intp[::1] inds=None):
    """Estimate the asymptotic standard deviation (uncertainty) in the stationary
    distribution, `\pi`.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
        The free parameters of the model. These values are the (possibly sparse)
        linearized elements of the log of the  upper triangular portion of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    n : int
        The size of `counts`
    inds : array, optional (default=None)
        Sparse linearized triu indices theta. If not supplied, theta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(theta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `theta` correspond.

    Returns
    -------
    sigma_pi : array, shape=(n,)
        Estimate of the element-wise asymptotic standard deviation of the stationary
        distribution, \pi.
    """
    cdef npy_intp i, j
    cdef npy_intp n_S_triu = n*(n-1)/2
    cdef npy_intp size = theta.shape[0]
    if not (inds is None or inds.shape[0] >= n):
        raise ValueError('inds must be None (dense) or an array longer than n')
    if inds is not None:
        if not np.all(inds == np.unique(inds)):
            raise ValueError('inds must be sorted, without redundant')
    if not ((theta.shape[0] == inds.shape[0]) or
            (inds is None and theta.shape[0] == n_S_triu + n)):
        raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
    if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
        raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('counts must be `size` x `size`')

    cdef double[::1] pi = zeros(n)
    cdef double[::1] temp = zeros(n)
    cdef double[::1] sigma_pi = zeros(n)
    cdef double[:, ::1] C_block = zeros((n, n))
    cdef double[:, ::1] dpi_dtheta = zeros((n, n))
    cdef double z = 0, pi_i = 0, z_m2 = 0, var_pi_i = 0

    # z = sum(pi)
    for i in range(n):
        pi_i = exp(theta[size-n+i])
        z += pi_i
        pi[i] = pi_i

    # z^{-2}
    z_m2 = 1.0 / (z * z)

    # copy the lower-right (n,n) block of covar_theta into contiguous memory
    # so that we can use BLAS
    for i in range(n):
        for j in range(n):
            C_block[i,j] = covar_theta[size-n+i, size-n+j]

    # build the Jacobian, \frac{d\pi}{d\theta}
    for i in range(n):
        for j in range(n):
            if i == j:
                dpi_dtheta[i, i] = z_m2 * pi[i] * (z-pi[i])
            else:
                dpi_dtheta[i, j] = -z_m2 * pi[i] * pi[j]

    # multiply in the Jacobian with the covariance matrix
    #\sigma_i = (h^i)^T M_{uv} (h^i)
    for i in range(n):
        cdgemv_N(C_block, dpi_dtheta[i], temp)
        cddot(dpi_dtheta[i], temp, &var_pi_i)
        sigma_pi[i] = sqrt(var_pi_i)

    return np.asarray(sigma_pi)



def sigma_timescales(const double[:, :] covar_theta, const double[::1] theta,
                           npy_intp n, npy_intp[::1] inds=None):
    """Estimate the asymptotic standard deviation (uncertainty) in the
    implied timescales.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
        The free parameters of the model. These values are the (possibly sparse)
        linearized elements of the log of the  upper triangular portion of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    n : int
        The size of `counts`
    inds : array, optional (default=None)
        Sparse linearized triu indices theta. If not supplied, theta is
        assumed to be a dense parameterization of the upper triangular portion
        of the symmetric rate matrix followed by the log equilibrium weights,
        and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
        set of indices, with  `len(inds) == len(theta)`,
        `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
        of the upper triangular elements of the rate matrix to which
        `theta` correspond.

    Returns
    -------
    sigma_t : array, shape=(n-1,)
        Estimate of the element-wise asymptotic standard deviation of the
        relaxation timescales of the model.
    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    cdef npy_intp u, v, i
    cdef double[::1] exptheta, var_T, w, dw_u, dw_v, temp, w_pow_m4, sigma
    cdef double[:, ::1] dKu, dKv, K, eye, AL, AR
    cdef npy_intp size = theta.shape[0]
    if not (inds is None or inds.shape[0] >= n):
        raise ValueError('inds must be None (dense) or an array longer than n')
    if inds is not None:
        if not np.all(inds == np.unique(inds)):
            raise ValueError('inds must be sorted, without redundant')
    if not ((theta.shape[0] == inds.shape[0]) or
            (inds is None and theta.shape[0] == n_S_triu + n)):
        raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
    if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
        raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('covar_theta must be `size` x `size`')

    var_T = zeros(n)
    dKu = zeros((n, n))
    dKv = zeros((n, n))
    dw_u = zeros(n)
    dw_v = zeros(n)
    w_pow_m4 = zeros(n)
    temp = zeros(n)
    S = zeros((n, n))
    exptheta = np.exp(theta)
    eye = np.eye(n)

    build_ratemat(exptheta, n, inds, S, 'S')
    w, U, V = eigK(S, n, exptheta[size-n:], 'S')

    order = np.argsort(w)[::-1]

    U = ascontiguousarray(np.asarray(U)[:, order])
    V = ascontiguousarray(np.asarray(V)[:, order])
    w = np.asarray(w)[order]

    for i in range(n):
        w_pow_m4[i] = w[i]**(-4)

    for u in range(size):
        dK_dtheta_A(exptheta, n, u, inds, None, dKu)
        dw_du(dKu, U, V, n, temp, dw_u)
        for v in range(size):
            dK_dtheta_A(exptheta, n, v, inds, None, dKv)
            dw_du(dKv, U, V, n, temp, dw_v)

            for i in range(n):
                var_T[i] += w_pow_m4[i] * dw_u[i] * dw_v[i] * covar_theta[u, v]

    # skip the stationary eigenvector
    sigma = zeros(n-1)
    for i in range(n-1):
        sigma[i] = sqrt(var_T[1+i])
    return np.asarray(sigma)
