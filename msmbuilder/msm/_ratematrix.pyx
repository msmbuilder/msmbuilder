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
include "_ratematrix_support.pyx"
include "_ratematrix_priors.pyx"


cpdef int build_ratemat(const double[::1] theta, npy_intp n, double[:, ::1] out,
                        const char* which=b'K'):
    r"""build_ratemat(theta, n, out, which='K')

    Build the reversible rate matrix K or symmetric rate matrix, S,
    from the free parameters, `\theta`

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`. These values are the linearized elements
        of the upper triangular portion of the symmetric rate matrix, S,
        followed by the log equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    which : {'S', 'K'}
        Whether to build the matrix S or the matrix K
    out : [output], array shape=(n, n)
        On exit, out contains the matrix K or S
    """
    cdef npy_intp u = 0, k = 0, i = 0, j = 0
    cdef npy_intp n_triu = n*(n-1)/2
    cdef double s_ij, K_ij, K_ji
    cdef double[::1] pi
    cdef int buildS = strcmp(which, 'S') == 0
    if DEBUG:
        assert out.shape[0] == n
        assert out.shape[1] == n
        assert theta.shape[0] == n_triu + n
        assert np.all(np.asarray(out) == 0)

    pi = zeros(n)
    for i in range(n):
        pi[i] = exp(theta[n_triu+i])

    for u in range(n_triu):
        k_to_ij(u, n, &i, &j)
        s_ij = theta[u]

        if DEBUG:
            assert 0 <= u < n*(n-1)/2

        K_ij = s_ij * sqrt(pi[j] / pi[i])
        K_ji = s_ij * sqrt(pi[i] / pi[j])
        if buildS:
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


cpdef double dK_dtheta_ij(const double[::1] theta, npy_intp n, npy_intp u,
                          double[:, ::1] A=None, double[:, ::1] out=None) nogil:
    r"""dK_dtheta_ij(theta, n, u, A=None, out=None)

    Compute :math:`dK_ij / dtheta_u` over all `i`, `j` for fixed `u`.

    Along with `dK_dtheta_u`, this function computes a slice of the 3-index
    tensor :math:`dK_ij / dtheta_u`, the derivative of the rate matrix `K`
    with respect to the free parameters,`\theta`. This function computes a 2D
    slice of this tensor over all (i,j) for a fixed `u`.

    Furthermore, this function _additionally_ makes it possible, using the
    argument `A`, to compute the hadamard product of this slice with a given
    matrix A directly.  Since dK/dtheta_u is a sparse matrix with a known
    sparsity structure, it's more efficient to just do the hadamard as we
    construct it, and never save the matrix elements directly.

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`. These values are the linearized elements
        of the upper triangular portion of the symmetric rate matrix, S,
        followed by the log equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    u : int
        The index, `0 <= u < len(theta)` of the element in `theta` to
        construct the derivative of the rate matrix, `K` with respect to.
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
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp a, i, j
    cdef double dK_i, s_ij, dK_ij, dK_ji, pi_i, pi_j
    cdef double sum_elem_product = 0
    if DEBUG:
        assert out.shape[0] == n and out.shape[1] == n
        assert A.shape[0] == n and A.shape[1] == n
        assert theta.shape[0] == n_triu + n

    if out is not None:
        memset(&out[0,0], 0, n*n*sizeof(double))

    if u < n_triu:
        # the perturbation is to the triu rate matrix
        # first, use the linear index, u, to get the (i,j)
        # indices of the symmetric rate matrix
        k_to_ij(u, n, &i, &j)

        s_ij = theta[u]
        pi_i = exp(theta[n_triu+i])
        pi_j = exp(theta[n_triu+j])
        dK_ij = sqrt(pi_j / pi_i)
        dK_ji = sqrt(pi_i / pi_j)

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
        pi_i = exp(theta[n_triu+i])

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
            s_ij = theta[k]
            pi_j = exp(theta[n_triu+j])
            dK_ij = -0.5 * s_ij * sqrt(pi_j / pi_i)
            dK_ji = 0.5  * s_ij * sqrt(pi_i / pi_j)

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


cpdef int dK_dtheta_u(const double[::1] theta, npy_intp n, npy_intp i,
                      npy_intp j, double[:] out, const double[:, :] A=None,
                      double[:] out2=None) nogil:
    r"""
    Compute :math:`dK_ij / dtheta_u` over all `u` for fixed (`i`, `j`).

    Along with `dK_dtheta_ij`, this function computes a slice of the 3-index
    tensor :math:`dK_ij / dtheta_u`, the derivative of the rate matrix `K`
    with respect to the free parameters,`\theta`. This function computes a 1D
    slice of this tensor over `u` for a fixed `i`, `j`, a gradient vector.

    Furthermore, this function _additionally_ makes it possible, using the
    argument `A`, to compute the vector-matrix product of this gradient vector
    with a given matrix A directly. Since the gradient is a sparse vector with
    a known sparsity structure, it's more efficient to just do the product as we
    construct it, and never save the elements directly.

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`. These values are the linearized elements
        of the upper triangular portion of the symmetric rate matrix, S,
        followed by the log equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    i, j : int
        The fixed indices of the rate matrix, `K`.
    out : [output], array of shape=(len(theta),)
        The vector dK[i,j]/dtheta[u] for all u is written here on exit
    A : array of shape(len(theta), len(theta)), optional
        If not None, an arbitrary matrix
    out2: [output], array of shape=(len(theta),)
        If not None, a second output where the product of `out` with `A`
        is written, ``out2 = np.dot(out, A)``.

    """
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp u, jj, size
    cdef double dK_i, dK_ij, dK_ji, pi_i, pi_j, pi_jj, val, sqrt_pi_j_over_i
    cdef double s_ij = 0
    size = n_triu + n
    cdef int compute_out2 = (A is not None and out2 is not None)

    if DEBUG:
        assert theta.shape[0] == size
        assert out.shape[0] == size
        assert out2 is None or out2.shape[0] == size

    # For u in 0...n_triu-1
    u = ij_to_k(i, j, n)
    pi_i = exp(theta[n_triu+i])
    pi_j = exp(theta[n_triu+j])

    if i != j:
        s_ij = theta[u]
        sqrt_pi_j_over_i = sqrt(pi_j / pi_i)
        out[u] += sqrt_pi_j_over_i
        if compute_out2:
            cdaxpy(sqrt_pi_j_over_i, x=A[u, :], y=out2)

        # for u in n_triu...size-1
        if s_ij != 0:
            val = 0.5 * s_ij * sqrt_pi_j_over_i
            out[n_triu + i] -= val
            out[n_triu + j] += val
            if compute_out2:
                cdaxpy(-val, x=A[n_triu + i, :], y=out2)
                cdaxpy(val, x=A[n_triu + j, :], y=out2)

    else:
        for jj in range(n):
            if jj == i:
                continue

            u = ij_to_k(i, jj, n)
            s_ij = theta[u]

            pi_jj = exp(theta[n_triu+jj])
            sqrt_pi_j_over_i = sqrt(pi_jj / pi_i)

            out[u] -= sqrt_pi_j_over_i
            if compute_out2:
                cdaxpy(-sqrt_pi_j_over_i, x=A[u, :], y=out2)

            if s_ij != 0:
                val = 0.5 * s_ij * sqrt_pi_j_over_i
                out[n_triu + i] += val
                out[n_triu + jj] -= val
                if compute_out2:
                    cdaxpy(val, x=A[n_triu + i, :], y=out2)
                    cdaxpy(-val, x=A[n_triu + jj, :], y=out2)

    return 0

def loglikelihood(const double[::1] theta, const double[:, ::1] counts, double t=1):
    r"""loglikelihood(theta, counts, n, t=1)

    Log likelihood and gradient of the log likelihood of a continuous-time
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
    t : double
        The lag time.

    Returns
    -------
    logl : double
        The log likelihood of the parameters.
    grad : array of shape = (n*(n-1)/2 + n)
        The gradient of the log-likelihood with respect to `\theta`
    """
    cdef npy_intp n = counts.shape[0]
    cdef npy_intp n_triu = n*(n-1)/2
    if not (counts.shape[0] == n and counts.shape[1] == n):
        raise ValueError('counts must be n x n')
    if not theta.shape[0] == n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')

    cdef npy_intp u, i, j
    cdef npy_intp size = theta.shape[0]
    cdef double logl = 0
    cdef double[::1] grad, w, pi
    cdef double[:, ::1] K, T, dT, U, V

    grad = zeros(size)
    S = zeros((n, n))
    T = zeros((n, n))
    dT = zeros((n, n))

    build_ratemat(theta, n, S, 'S')
    if not np.all(np.isfinite(S)):
        # these parameters don't seem good...
        # tell the optimizer to stear clear!
        return np.nan, ascontiguousarray(grad)

    pi = zeros(n)
    for i in range(n):
        pi[i] = exp(theta[n_triu+i])

    w, U, V = eig_K(S, n, pi, 'S')
    dT_dtheta(w, U, V, counts, n, t, T, dT)

    with nogil:
        for u in range(size):
            grad[u] = dK_dtheta_ij(theta, n, u, A=dT)

        for i in range(n):
            for j in range(n):
                if counts[i, j] > 0:
                    logl += counts[i, j] * log(T[i, j])

    return logl, ascontiguousarray(grad)


def hessian(double[::1] theta, double[:, ::1] counts, double t=1, npy_intp[::1] inds=None):
    r"""hessian(theta, counts, t=1)

    Estimate of the hessian of the log-likelihood with respect to \theta.

    Parameters
    ----------
    theta : array of shape = (n*(n-1)/2 + n) for dense or shorter
        The free parameters of the model. These values are the (possibly sparse)
        linearized elements of the log of the  upper triangular portion of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    counts : array of shape = (n, n)
        The matrix of observed transition counts.
    inds : array of ints or None
        If supplied, only compute a block of the Hessian at the specified
        indices.
    t : double
        The lag time.

    Notes
    -----
    This computation follows equation 3.6 of [1].

    References
    ----------
    .. [1] Kalbfleisch, J. D., and Jerald F. Lawless. "The analysis of panel data
       under a Markov assumption." J. Am. Stat. Assoc. 80.392 (1985): 863-871.

    Returns
    -------
    m : array, shape=(len(theta), len(theta))
        An estimate of the hessian of the log-likelihood
    """
    cdef npy_intp n = counts.shape[0]
    cdef npy_intp n_triu = n*(n-1)/2
    if not (counts.shape[0] == n and counts.shape[1] == n):
        raise ValueError('counts must be n x n')
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')

    cdef npy_intp size = theta.shape[0]
    cdef npy_intp u, uu, v, vv, i, j
    cdef double hessian_uv
    cdef double[::1] grad, pi, expwt
    cdef double[:, ::1] K, T, Q, dKu,  Au, temp1, temp2

    if inds is None:
        inds = np.arange(size, dtype=np.intp)
        hessian = zeros((size, size))
    else:
        hessian = zeros((len(inds), len(inds)))

    pi = zeros(n)
    expwt = zeros(n)
    S = zeros((n, n))
    T = zeros((n, n))
    Q = zeros((n, n))
    dKu = zeros((n, n))
    Au = zeros((n, n))
    temp1 = zeros((n, n))
    temp2 = zeros((n, n))
    rowsums = np.sum(counts, axis=1)
    transtheta = zeros(size)

    for i in range(n):
        pi[i] = exp(theta[n_triu+i])

    build_ratemat(theta, n, S, 'S')
    w, U, V = eig_K(S, n, pi, 'S')

    for i in range(n):
        expwt[i] = exp(w[i]*t)

    transmat(expwt, U, V, n, temp1, T)  # write transmat into T

    for i in range(n):
        for j in range(n):
            Q[i,j] = -rowsums[i] / T[i, j]

    for uu in range(len(inds)):
        u = inds[uu]

        dK_dtheta_ij(theta, n, u, None, out=dKu)
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

        for vv in range(uu, len(inds)):
            v = inds[vv]

            hessian_uv = dK_dtheta_ij(theta, n, v, A=Au)
            hessian[uu, vv] = hessian_uv
            hessian[vv, uu] = hessian_uv

    return np.asarray(hessian)


def sigma_K(const double[:, ::1] covar_theta, const double[::1] theta, npy_intp n):
    r"""sigma_K(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty in the rate
    matrix, `K`

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n)
        The free parameters of the model at the MLE. These values are the
        linearized elements of the upper triangular portion of the symmetric
        rate matrix, S, followed by the log of the equilibrium distribution.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_K : array, shape=(n, n)
        Estimate of the element-wise asymptotic standard deviation of the
        rate matrix, K.
    """
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp u, v, i, j
    cdef double var_K_ij
    cdef double[::1] temp, dKij
    cdef double[:, ::1] sigma_K
    cdef npy_intp size = theta.shape[0]
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('covar_theta must be `size` x `size`')

    sigma_K = zeros((n, n))
    dKij = zeros(size)
    temp = zeros(size)

    for i in range(n):
        for j in range(n):
            # dK_dtheta_u(theta, n, i, j, out=dKij)
            # cdgemv_N(covar_theta, dKij, temp)

            memset(&dKij[0], 0, size*sizeof(double))
            memset(&temp[0], 0, size*sizeof(double))
            dK_dtheta_u(theta, n, i, j, out=dKij, A=covar_theta, out2=temp)
            cddot(dKij, temp, &var_K_ij)
            sigma_K[i, j] = sqrt(var_K_ij)

    return np.asarray(sigma_K)


def sigma_pi(const double[:, :] covar_theta, const double[::1] theta, npy_intp n):
    r"""sigma_pi(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty) in the stationary
    distribution, `\pi`.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n)
        The free parameters of the model at the MLE. These values are the
        linearized elements of the upper triangular portion of the symmetric
        rate matrix, S, followed by the log of the equilibrium distribution.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_pi : array, shape=(n,)
        Estimate of the element-wise asymptotic standard deviation of the stationary
        distribution, \pi.
    """
    cdef npy_intp i, j
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp size = theta.shape[0]
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('covar_theta must be `size` x `size`')

    cdef double[::1] pi = zeros(n)
    cdef double[::1] temp = zeros(n)
    cdef double[::1] sigma_pi = zeros(n)
    cdef double[:, ::1] C_block = zeros((n, n))
    cdef double[:, ::1] dpi_dtheta = zeros((n, n))
    cdef double z = 0, pi_i = 0, z_m2 = 0, var_pi_i = 0

    # z = sum(pi)
    for i in range(n):
        pi_i = exp(theta[n_triu+i])
        z += pi_i
        pi[i] = pi_i

    # z^{-2}
    z_m2 = 1.0 / (z * z)

    # copy the lower-right (n,n) block of covar_theta into contiguous memory
    # so that we can use BLAS
    for i in range(n):
        for j in range(n):
            C_block[i,j] = covar_theta[n_triu+i, n_triu+j]

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


def sigma_eigenvalues(const double[:, ::1] covar_theta, const double[::1] theta,
                     npy_intp n):
    r"""sigma_eigenvalues(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty) in the
    eigenvalues of K

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n)
        The free parameters of the model at the MLE. These values are the
        linearized elements of the upper triangular portion of the symmetric
        rate matrix, S, followed by the log of the equilibrium distribution.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_eigenvalues : array, shape=(n,)
        Estimate of the element-wise asymptotic standard deviation of the
        eigenvalues of the rate matrix (in sorted order).
    """
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp u, v, i
    cdef double var_w_i
    cdef double[::1] pi, sigma_w, temp1, temp2
    cdef double[:, ::1] S, U, V, dKu
    cdef double[::1, :] dlambda_dtheta
    cdef npy_intp size = theta.shape[0]
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('covar_theta must be `size` x `size`')

    pi = zeros(n)
    S = zeros((n, n))
    dKu = zeros((n, n))
    temp1 = zeros(n)
    temp2 = zeros(size)
    sigma_w = zeros(n)
    dlambda_dtheta = np.zeros((n, size), order='F')

    for i in range(n):
        pi[i] = exp(theta[n_triu+i])

    build_ratemat(theta, n, S, 'S')
    w, U, V = eig_K(S, n, pi, 'S')

    order = np.argsort(w)[::-1]

    U = ascontiguousarray(np.asarray(U)[:, order])
    V = ascontiguousarray(np.asarray(V)[:, order])
    w = np.asarray(w)[order]

    for u in range(size):
        dK_dtheta_ij(theta, n, u, A=None, out=dKu)
        dw_du(dKu, U, V, n, temp1, dlambda_dtheta[:, u])

    for i in range(n):
        cdgemv_N(covar_theta, dlambda_dtheta[i, :].copy(), temp2)
        cddot(dlambda_dtheta[i, :], temp2, &var_w_i)
        sigma_w[i] = sqrt(var_w_i)

    return np.asarray(sigma_w)


def sigma_timescales(const double[:, ::1] covar_theta, const double[::1] theta,
                     npy_intp n):
    r"""sigma_timescales(covar_theta, theta, n):

    Estimate the asymptotic standard deviation (uncertainty) in the
    implied timescales.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)/2 + n)
        The free parameters of the model at the MLE. These values are the
        linearized elements of the upper triangular portion of the symmetric
        rate matrix, S, followed by the log of the equilibrium distribution.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_t : array, shape=(n-1,)
        Estimate of the element-wise asymptotic standard deviation of the
        relaxation timescales of the model.
    """
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp u, v, i
    cdef double var_tau_i, d_lambda_i_d_theta_u
    cdef double[::1] pi, w, sigma_tau, dw_u, temp1, temp2
    cdef double[:, ::1] S, U, V, dKu, dtau_dtheta
    cdef double[::1, :] dlambda_dtheta
    cdef npy_intp size = theta.shape[0]
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    if not covar_theta.shape[0] == size and covar_theta.shape[1] == size:
        raise ValueError('covar_theta must be `size` x `size`')

    pi = zeros(n)
    S = zeros((n, n))
    dKu = zeros((n, n))
    temp1 = zeros(n)
    temp2 = zeros(size)
    sigma_tau = zeros(n-1)
    dlambda_dtheta = np.zeros((n, size), order='F')
    dtau_dtheta = np.zeros((n, size))

    for i in range(n):
        pi[i] = exp(theta[n_triu+i])

    build_ratemat(theta, n, S, 'S')
    w, U, V = eig_K(S, n, pi, 'S')

    order = np.argsort(w)[::-1]
    U = ascontiguousarray(np.asarray(U)[:, order])
    V = ascontiguousarray(np.asarray(V)[:, order])
    w = np.asarray(w)[order]

    for u in range(size):
        dK_dtheta_ij(theta, n, u, A=None, out=dKu)
        dw_du(dKu, U, V, n, temp1, dlambda_dtheta[:, u])
        for i in range(n):
            dtau_dtheta[i, u] = dlambda_dtheta[i, u] / (w[i]**2)

    for i in range(1, n):
        cdgemv_N(covar_theta, dtau_dtheta[i], temp2)
        cddot(dtau_dtheta[i], temp2, &var_tau_i)
        sigma_tau[i-1] = sqrt(var_tau_i)

    return np.asarray(sigma_tau)
