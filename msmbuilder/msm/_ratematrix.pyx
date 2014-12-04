# cython: boundscheck=False, cdivision=True, wraparound=False
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
from numpy import (zeros, allclose, array, real, ascontiguousarray, dot, diag)
import scipy.linalg
from scipy.linalg import blas, eig
from numpy cimport npy_intp
from libc.math cimport sqrt, log, exp
from libc.string cimport memset
from libc.stdlib cimport calloc, malloc, free
from cython.parallel cimport prange, parallel
from cython.operator cimport dereference as deref

include "cy_blas.pyx"
include "config.pxi"
include "triu_utils.pyx"      # ij_to_k() and k_to_ij()
include "binary_search.pyx"   # bsearch()
IF OPENMP:
    cimport openmp


cpdef int buildK(const double[::1] exptheta, npy_intp n, const npy_intp[::1] inds,
                 double[:, ::1] out):
    """Build the reversible rate matrix K from the free parameters, `\theta`

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
    Notes
    -----
    The last `n` elements of exptheta must be nonzero, since they parameterize
    the equilibrium populations, so even with the sparse parameterization,
    `len(u)` must be greater than or equal to `n`.

    With the sparse parameterization, the following invariant holds. If

        inds = indices_of_nonzero_elements(exptheta)

    Then,

        buildK(exptheta, n, None) == buildK(exptheta[inds], n, inds)

    Returns
    -------
    out : [output], 2d array of shape = (n, n)
        The rate matrix is written into this array

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


cpdef double dK_dtheta_A_sum(const double[::1] exptheta, npy_intp n, npy_intp u,
                             const npy_intp[::1] inds, const double[:, ::1] A) nogil:
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
    A : array of shape=(n, n)
        An arbitrary (n, n) matrix to be multiplied elementwise
        with the derivative of the rate matrix.

    Returns
    -------
    s : double
        The sum of the element-wise product of dK/du and A
    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    cdef npy_intp a, i, j, n_triu, uu, kk
    cdef double dK_i, s_ij, dK_ij, dK_ji
    cdef double sum_elem_product = 0
    cdef double[::1] pi
    cdef double* dK_ii
    # dK_ii = <double*> calloc(n, sizeof(double))
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

        sum_elem_product = (
            A[i,j]*dK_ij +  A[j,i]*dK_ji
          - A[i,i]*dK_ij - A[j,j]*dK_ji
        )

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
            sum_elem_product += (
                A[i,j]*dK_ij +  A[j,i]*dK_ji
                - A[i,i]*dK_ij - A[j,j]*dK_ji
            )

    return sum_elem_product


cdef dT_dtheta(const double[:, ::1] K, const double[:, ::1] counts,
               npy_intp n, double t,
               double[:, ::1] T, double[:, ::1] dT):
    """Compute some of the terms required for d(exp(K))/d(theta).

    Returns
    -------
    """
    cdef npy_intp i, j
    cdef double norm
    cdef double[::1] w, expwt
    cdef double[:, ::1] U, V, X, temp1, temp2, dLdK
    temp1 = zeros((n, n))
    temp2 = zeros((n, n))
    dLdK = zeros((n, n))
    expwt = zeros(n)

    w_, U_, V_ = scipy.linalg.eig(K, left=True, right=True)
    U = ascontiguousarray(real(U_))
    V = ascontiguousarray(real(V_))
    w = ascontiguousarray(real(w_))

    with nogil:
        for i in range(n):
            # we need to ensure the proper normalization
            cddot(U[:, i], V[:, i], &norm)
            for j in range(n):
                U[j, i] = U[j, i] / norm

    if DEBUG:
        assert np.allclose(scipy.linalg.inv(V).T, U)

    with nogil:
        for i in range(n):
            expwt[i] = exp(w[i]*t)

        # transmat = np.dot(np.dot(V, np.diag(expwt)), U.T)
        for i in range(n):
            for j in range(n):
                temp1[i, j] = V[i, j] * expwt[j]
        cdgemm_NT(temp1, U, T)

        # dLdK[i,j] = counts[i,j] / T[i,j]
        for i in range(n):
            for j in range(n):
                dLdK[i, j] = counts[i, j] / T[i, j]

        # out = U \left(V^T dLdK U \circ X(\lambda, t))\right) V^T

        # temp2 = V^T dLdK U
        cdgemm_TN(V, dLdK, temp1)
        cdgemm_NN(temp1, U, temp2)

        # temp2 =  (V^T dLdK U \circ X(w, t))
        for i in range(n):
            for j in range(n):
                if i != j:
                    temp2[i, j] *= (expwt[i] - expwt[j]) / (w[i] - w[j])
                else:
                    temp2[i, j] *= t * expwt[i]

        # dT = U \left(V^TCU \circ X(\lambda, t))\right) V^T
        cdgemm_NN(U, temp2, temp1)
        cdgemm_NT(temp1, V, dT)

    if DEBUG:
        X = np.subtract.outer(expwt, expwt) / np.subtract.outer(w, w)
        np.fill_diagonal(np.asarray(X), t*np.asarray(expwt))
        Y = np.asarray(U).dot(np.asarray(V).T.dot(dLdK).dot(U) * X).dot(np.asarray(V).T)

        assert np.allclose(dT, Y)
        assert np.allclose(T, np.dot(np.dot(V, np.diag(expwt)), U.T))
        assert np.allclose(T, scipy.linalg.expm(t*np.asarray(K)))


def loglikelihood(const double[::1] theta, const double[:, ::1] counts, npy_intp n,
                  const npy_intp[::1] inds=None, double t=1, npy_intp n_threads=1):
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
    n_threads : int
        The number of threads to use in parallel.

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
    cdef double[::1] grad, exptheta
    cdef double[:, ::1] K, T, dT

    grad = zeros(size)
    exptheta = zeros(size)
    K = zeros((n, n))
    T = zeros((n, n))
    dT = zeros((n, n))

    for u in range(size):
        exptheta[u] = exp(theta[u])

    buildK(exptheta, n, inds, K)
    if not np.all(np.isfinite(K)):
        # these parameters don't seem good...
        # tell the optimizer to stear clear!
        return -np.inf, ascontiguousarray(grad)

    dT_dtheta(K, counts, n, t, T, dT)

    with nogil:
        for u in range(size):
            grad[u] = dK_dtheta_A_sum(exptheta, n, u, inds, dT)

        for i in range(n):
            for j in range(n):
                logl += counts[i, j] * log(T[i, j])    

    return logl, ascontiguousarray(grad)


def hessian(double[::1] theta, double[:, ::1] counts, npy_intp n, double t=1,
            npy_intp[::1] inds=None, npy_intp n_threads=1):
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
    n_threads : int
        The number of threads to use in parallel.

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
    cdef double[::1] grad, exptheta
    cdef double[:, ::1] K, T, dT

    grad = zeros(size)
    exptheta = zeros(size)
    K = zeros((n, n))
    T = zeros((n, n))
    dT = zeros((n, n))
    
    buildK(exptheta, n, inds, K)
    dT_dtheta(K, counts, n, t, T, dT)


    '''
    cdef int thread_num = 0
    cdef double logl
    cdef double[::1] w, expwt, exptheta, rowsums, hessian_uv
    cdef double[:, ::1] K, transmat, AL, AR, temp, hessian
    cdef double[:, :, ::1] dPu, dPv
    cdef double[:, :, :, ::1] temp1

    exptheta = np.exp(theta)
    K = zeros((n, n))
    temp = zeros((n, n))
    temp1 = zeros((n_threads, 2, n, n))
    dPu = zeros((n_threads, n, n))
    dPv = zeros((n_threads, n, n))
    hessian = zeros((size, size))
    hessian_uv = zeros(n_threads)
    rowsums = np.sum(counts, axis=1)

    buildK(exptheta, n, inds, K)
    AL, AR, w, expwt = dP_dtheta_terms(K, n, t)

    transmat = np.dot(np.dot(AR, np.diag(expwt)), AL.T)

    with nogil, parallel(num_threads=n_threads):
        IF OPENMP:
            thread_num = openmp.omp_get_thread_num()
        for u in range(size):
            # write dP / dtheta_u into dPu[thread_num]
            build_dPu(AL, AR, expwt, w, exptheta, n, u, t, inds, temp1[thread_num, 0], temp1[thread_num, 1], dPu[thread_num])

            for v in range(size):
                # write dP / dtheta_v into dPv[thread_num]
                build_dPu(AL, AR, expwt, w, exptheta, n, v, t, inds, temp1[thread_num, 0], temp1[thread_num, 1], dPv[thread_num])

                hessian_uv[thread_num] = 0
                for i in range(n):
                    for j in range(n):
                        hessian_uv[thread_num] += (rowsums[i]/transmat[i,j]) * (dPu[thread_num, i, j] * dPv[thread_num, i, j])
                hessian[u, v] = hessian_uv[thread_num]
    '''
    return np.asarray(hessian)


# def uncertainty_K(const double[:, :] invhessian, const double[::1] theta,
#                   npy_intp n, npy_intp[::1] inds=None, npy_intp n_threads=1):
#     """Estimate of the uncertainty in the rate matrix, `K`
#
#     Parameters
#     ----------
#     invhessian : array, shape=(len(theta), len(theta))
#         Inverse of the hessian of the log-likelihood
#     theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
#         The free parameters of the model. These values are the (possibly sparse)
#         linearized elements of the log of the  upper triangular portion of the
#         symmetric rate matrix, S, followed by the log of the equilibrium
#         distribution.
#     n : int
#         The size of `counts`
#     inds : array, optional (default=None)
#         Sparse linearized triu indices theta. If not supplied, theta is
#         assumed to be a dense parameterization of the upper triangular portion
#         of the symmetric rate matrix followed by the log equilibrium weights,
#         and must be of length `n*(n-1)/2 + n`. If `inds` is supplied, it is a
#         set of indices, with  `len(inds) == len(theta)`,
#         `0 <= inds < n*(n-1)/2+n`, giving the indices of the nonzero elements
#         of the upper triangular elements of the rate matrix to which
#         `theta` correspond.
#     n_threads : int
#         The number of threads to use in parallel.
#
#     Returns
#     -------
#     sigma_K : array, shape=(n, n)
#         Estimate of the element-wise asymptotic standard deviation of the rate matrix, K.
#     """
#     cdef npy_intp n_S_triu = n*(n-1)/2
#     cdef npy_intp u, v, i, j
#     cdef double[::1] exptheta
#     cdef double[:, ::1] var_K, dKu, dKv, K, eye
#     cdef npy_intp size = theta.shape[0]
#     if not (inds is None or inds.shape[0] >= n):
#         raise ValueError('inds must be None (dense) or an array longer than n')
#     if inds is not None:
#         if not np.all(inds == np.unique(inds)):
#             raise ValueError('inds must be sorted, without redundant')
#     if not ((theta.shape[0] == inds.shape[0]) or
#             (inds is None and theta.shape[0] == n_S_triu + n)):
#         raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
#     if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
#         raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')
#     if not invhessian.shape[0] == size and invhessian.shape[1] == size:
#         raise ValueError('counts must be `size` x `size`')
#
#     var_K = zeros((n, n))
#     dKu = zeros((n, n))
#     dKv = zeros((n, n))
#     K = zeros((n, n))
#     exptheta = np.exp(theta)
#     eye = np.eye(n)
#
#     buildK(exptheta, n, inds, K)
#
#     for u in range(size):
#         dK_dtheta_A(exptheta, n, u, inds, eye, dKu)
#         for v in range(size):
#             dK_dtheta_A(exptheta, n, v, inds, eye, dKv)
#             # this could be optimized, since dKu and dKv are sparse and we
#             # know their pattern
#             for i in range(n):
#                 for j in range(n):
#                     var_K[i,j] += invhessian[u,v] * dKu[i,j] * dKv[i,j]
#
#     return np.asarray(np.sqrt(var_K))


def uncertainty_pi(const double[:, :] invhessian, const double[::1] theta,
                   npy_intp n, npy_intp[::1] inds=None):
    """Estimate of the uncertainty in the stationary distribution, `\pi`.

    Parameters
    ----------
    invhessian : array, shape=(len(theta), len(theta))
        Inverse of the hessian of the log-likelihood
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
    cdef npy_intp i
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
    if not invhessian.shape[0] == size and invhessian.shape[1] == size:
        raise ValueError('counts must be `size` x `size`')

    cdef double[::1] pi = np.exp(theta)[size-n:]

    var_pi = zeros(n)
    for i in range(n):
        var_pi[i] = pi[i] * invhessian[size-n+i, size-n+i]

    return np.asarray(np.sqrt(var_pi))


cpdef int dw_du(const double[:, ::1] dKu, const double[:, ::1] AL,
            const double[:, ::1] AR, npy_intp n, double[::1] temp,
            double[::1] out) nogil:
    r"""Calculate the derivative of the eigenvalues, w, of a matrix, K(\theta),
    with respect to \theta_u.

    Parameters
    ----------
    dKu : array, shape=(n, n)
        Derivative of the rate matrix, K(\theta), with respect to \theta_u
    AL : array, shape=(n, n)
        Left eigenvectors of the rate matrix, K(\theta)
    AR : array, shape=(n, n)
        Right eigenvectors of the rate matrix, K(\theta)
    n : int
        Size of the matrices
    temp : array, shape=(n,)
        Temporary storage (overwritten)

    Returns
    -------
    out : array, shape=(n,)
        On exit, out[i] contains the derivative of the `i`th eigenvalue
        of K with respect to \theta_u.
    """
    cdef npy_intp i
    for i in range(n):
        cdgemv_N(dKu, AR[:, i], temp)
        cddot(temp, AL[:, i], &out[i])

#
# def uncertainty_timescales(const double[:, :] invhessian, const double[::1] theta,
#                            npy_intp n, npy_intp[::1] inds=None, npy_intp n_threads=1):
#     """
#
#     """
#     cdef npy_intp n_S_triu = n*(n-1)/2
#     cdef npy_intp u, v, i
#     cdef double[::1] exptheta, var_T, w, dw_u, dw_v, temp, w_pow_m4
#     cdef double[:, ::1] dKu, dKv, K, eye, AL, AR
#     cdef npy_intp size = theta.shape[0]
#     if not (inds is None or inds.shape[0] >= n):
#         raise ValueError('inds must be None (dense) or an array longer than n')
#     if inds is not None:
#         if not np.all(inds == np.unique(inds)):
#             raise ValueError('inds must be sorted, without redundant')
#     if not ((theta.shape[0] == inds.shape[0]) or
#             (inds is None and theta.shape[0] == n_S_triu + n)):
#         raise ValueError('theta must have shape n*(n+1)/2+n, or match inds')
#     if inds is not None and not np.all(inds[-n:] == n*(n-1)/2 + np.arange(n)):
#         raise ValueError('last n indices of inds must be n*(n-1)/2, ..., n*(n-1)/2+n-1')
#     if not invhessian.shape[0] == size and invhessian.shape[1] == size:
#         raise ValueError('counts must be `size` x `size`')
#
#     var_T = zeros(n)
#     dKu = zeros((n, n))
#     dKv = zeros((n, n))
#     dw_u = zeros(n)
#     dw_v = zeros(n)
#     w_pow_m4 = zeros(n)
#     temp = zeros(n)
#     K = zeros((n, n))
#     exptheta = np.exp(theta)
#     eye = np.eye(n)
#
#     buildK(exptheta, n, inds, K)
#     AL, AR, w, _ = dP_dtheta_terms(K, n, 1.0)
#     order = np.argsort(w)[::-1]
#     AL = ascontiguousarray(np.asarray(AL)[:, order])
#     AR = ascontiguousarray(np.asarray(AR)[:, order])
#     w = np.asarray(w)[order]
#
#     for i in range(n):
#         w_pow_m4[i] = w[i]**(-4)
#
#     for u in range(size):
#         dK_dtheta_A(exptheta, n, u, inds, eye, dKu)
#         dw_du(dKu, AL, AR, n, temp, dw_u)
#         for v in range(size):
#             dK_dtheta_A(exptheta, n, v, inds, eye, dKv)
#             dw_du(dKv, AL, AR, n, temp, dw_v)
#
#             for i in range(n):
#                 var_T[i] += w_pow_m4[i] * dw_u[i] * dw_v[i] * invhessian[u, v]
#
#     return np.asarray(np.sqrt(var_T))[1:]



def _supports_openmp():
    """Does the system support OpenMP?"""
    IF OPENMP:
        return True
    return False
