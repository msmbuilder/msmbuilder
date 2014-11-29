# cython: boundscheck=False, cdivision=True, wraparound=False
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
"""Implementation of the log-likelihood function and gradient for a
continuous-time reversible Markov model sampled at a regular interval.

Theory
------

Consider an `n`-state time-homogeneous Markov process, `X(t)` At time `t`, the
`n`-vector P(t) = `Pr(X(t) = i)` is probability that the system is in each of
the `n` states. These probabilities evolve forward in time, governed by
an `n x n` transition rate matrix `K` ::

    dP(t)/dt = P(t) * K

The solution is ::

    P(t) = exp(tK) * P(0)

In other words, the state-to-state lag-\tau transition probability is ::

    P[ X(t+\tau) = j | X(t) = i ] = exp(\tau K)_{ij}

For this model, we observe the evolution of one or more chains, `X(t)` at a
regular interval, `\tau`. Let `C_{ij}` be the number of times the chain was
observed at state `i` at time `t` and at state `j` at time `t+\tau` (the number
of observed transition counts). Suppose that `K` depends on a length-`b`
parameter vector, `\theta`. The log-likelihood is ::

  L(\theta) = \sum_{ij} C_{ij} log(exp(\tau K(\theta))_{ij})

The function :func:`loglikelihood` computes this log-likelihood and and gradient
of the log-likelihood with respect to `\theta`.

Parameterization
----------------
This code parameterizes K(\theta) such that K is constrained to satisfy
detailed-balance with respect to a stationary distribution, `pi`. For an
`n`-state model, `theta` is of length `n*(n-1)/2 + n`. The first `n*(n-1)/2`
elements of theta are the parameterize the symmetric rate matrix, S, and the
remaining `n` entries parameterize the stationary distribution. For n=3, the
function K(\theta) is shown below ::

  s_u   =  exp(\theta_u)                for  u = 0, 1, ..., n*(n-1)/2
  pi_i  =  exp(\theta_{i + n*(n-1)/2})  for  i = 0, 1, ..., n
  r_i   =  sqrt(pi_i)                   for  i = 0, 1, ..., n

     [[     k_00,     s0*(r0/r1),  s1*r(0/r2),  s2*(r0/r3)  ],
      [  s0*(r1/r0),     k_11,     s3*(r1/r2),  s4*(r1/r3)  ],
  K = [  s1*(r2/r0),  s3*(r2/r1),     k_22,     s5*(r2/r3)  ],
      [  s2*(r3/r0),  s4*(r3/r1),  s5*(r3/r2),      k_33    ]]

  where the diagonal elements `k_ii` are set such that the row sums of `K` are
  all zero, `k_{ii} = -\sum_{j != i} K_{ij}`

This form for `K` satisfies detailed balance by construction ::

  K_{ij} / K_{ji}  =  ri**2 / rj**2  =  pi_i / pi_j

Note that `K` is built from `exp(\theta)`. This parameterization makes
optimization easier, since it keeps the off-diagonal entries positive and the
diagonal entries negative.

Performance
-----------
This is pretty low-level cython code. Everything is explicitly typed, and
we use cython memoryviews. Using the wrappers from `cy_blas.pyx`, we make
direct calls to the FORTRAN BLAS matrix multiply using the function pointers
that scipy exposes.

There are assert statements which provide some clarity, but are slow.
They are compiled away by default using the CYTHON_WITHOUT_ASSERTIONS macro.
To enable them when compiling from source `python setup.py install` needs to
be run with the flag `--debug` on the command line.

References
----------
..[1] Kalbfleisch, J. D., and Jerald F. Lawless. "The analysis of panel data
under a Markov assumption." J. Am. Stat. Assoc. 80.392 (1985): 863-871.
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
IF OPENMP:
    cimport openmp


cdef inline npy_intp ij_to_k(npy_intp i, npy_intp j, npy_intp n) nogil:
    # (i, j) 2D index to linearized upper triaungular index
    # See also k_to_ij, the inverse operation
    if j > i:
        return (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
    return (n*(n-1)/2) - (n-j)*((n-j)-1)/2 + i - j - 1


cdef inline void k_to_ij(npy_intp k, npy_intp n, npy_intp *i, npy_intp *j) nogil:
    # linearized upper triangular index, k, to -> (i, j) 2D index

    # E.g with the 4x4 upper-triangular matrix, we need
    # to get the (i,j) index of an element from its
    # linear index:
    #  [ 0  a0  a1  a2  a3 ]      0 -> (i=0,j=1)
    #  [ 0   0  a4  a5  a6 ]      1 -> (i=0,j=2)
    #  [ 0   0   0  a7  a8 ]      5 -> (i=1,j=3)
    #  [ 0   0   0   0  a9 ]            etc
    #  [ 0   0   0   0   0 ]
    # http://stackoverflow.com/a/27088560/1079728

    i[0] = n - 2 - <int>(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j[0] = k + i[0] + 1 - n*(n-1)/2 + (n-i[0])*(n-i[0]-1)/2


cpdef inline npy_intp bsearch(npy_intp[::1] haystack, npy_intp needle) nogil:
    # http://rosettacode.org/wiki/Binary_search#Python:_Iterative
    cdef npy_intp mid
    cdef npy_intp low = 0
    cdef npy_intp high = haystack.shape[0] - 1
    # unsigned to avoid overflow in `(left + right)/2`

    while low <= high:
        mid = low + (high - low) / 2
        if haystack[mid] > needle:
            high = mid - 1
        elif haystack[mid] < needle:
            low = mid + 1
        else:
            return mid

    return -1


cpdef int buildK(double[::1] exptheta, npy_intp n, npy_intp[::1] inds,
                 double[:, ::1] out):
    """Build the reversible rate matrix K from the free parameters, `\theta`

    Parameters
    ----------
    exptheta : [input], array
        The element-wise exponential of the free parameters, `\theta`.
        These values are the linearized elements of the upper triangular portion
        of the symmetric rate matrix, S.
    n : [input]
        The dimension of the K matrix
    inds : [input, optional] (default=None)
        If not supplied, exptheta is assumed to be a dense parameterization of
        the upper triangular portion of the symmetric rate matrix, and must be
        of length `n*(n-1)/2`. If `inds` is supplied, it is a set of indices,
        with `len(inds) == len(exptheta)`, `0 <= inds < n*(n-1)/2`, giving the
        indices of the nonzero elements of the upper triangular elements of
        the rate matrix.
    out : [output], 2d array of shape = (n, n)
        The rate matrix is written into this array

    Notes
    -----
    The last `n` elements of exptheta must be nonzero, since they parameterize
    the equilibrium populations, so even with the sparse parameterization,
    `len(u)` must be greater than or equal to `n`.

    inds = indices_of_nonzero_elements(exptheta)
    buildK(exptheta, n, None) == buildK(exptheta[inds], n, inds)
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


cpdef int dK_dtheta_A(double[::1] exptheta, npy_intp n, npy_intp u, npy_intp[::1] inds,
                      double[:, ::1] A, double[:, ::1] out) nogil:
    """Compute the matrix product of the derivative of (the rate matrix, `K`,
    with respect to the free parameters,`\theta`, dK_ij / dtheta_u) and another
    matrix, A.

    Since dK/dtheta_u is a sparse matrix with a known sparsity structure, it's
    more efficient to just do the matrix multiply as we construct it, and never
    save the matrix elements directly.

    """
    cdef npy_intp n_S_triu = n*(n-1)/2
    cdef npy_intp a, i, j, n_triu, uu, kk
    cdef double dK_i, s_ij, dK_ij, dK_ji
    cdef double[::1] pi
    cdef double* dK_ii
    dK_ii = <double*> calloc(n, sizeof(double))
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
        dK_ii[i] = -dK_ij
        dK_ii[j] = -dK_ji

        # now, dK contains 4 nonzero elements, at
        #  (i, i), (i, j) (j, i) and (j, j), so the matrix*matrix product
        # populates two rows in `out`

        for a in range(n):
            out[i, a] = dK_ii[i] * A[i, a] + dK_ij * A[j, a]
            out[j, a] = dK_ii[j] * A[j, a] + dK_ji * A[i, a]
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
            dK_ii[i] -= dK_ij
            dK_ii[j] -= dK_ji

            for a in range(n):
                out[i, a] += dK_ij * A[j, a]
                out[j, a] += dK_ji * A[i, a]

        for i in range(n):
            dK_i = dK_ii[i]
            for j in range(n):
                out[i, j] += dK_i * A[i, j]

    free(dK_ii)


cdef dP_dtheta_terms(double[:, ::1] K, npy_intp n, double t):
    """Compute some of the terms required for d(exp(K))/d(theta). This
    includes the left and right eigenvectors of K, the eigenvalues, and
    the exp of the eigenvalues.

    Returns
    -------
    AL : array of shape = (n, n)
        The left eigenvectors of K
    AR : array of shape = (n, n)
        The right eigenvectors of K
    w : array of shape = (n,)
        The eigenvalues of K
    expw : array of shape = (n,)
        The exp of the eigenvalues of K
    """
    cdef npy_intp i
    w, AL, AR = scipy.linalg.eig(K, left=True, right=True)

    for i in range(n):
        # we need to ensure the proper normalization
        AL[:, i] /= dot(AL[:, i], AR[:, i])

    if DEBUG:
        assert np.allclose(scipy.linalg.inv(AR).T, AL)

    AL = ascontiguousarray(real(AL))
    AR = ascontiguousarray(real(AR))
    w = ascontiguousarray(real(w))
    expwt = zeros(w.shape[0])
    for i in range(w.shape[0]):
        expwt[i] = exp(w[i]*t)

    return AL, AR, w, expwt


cdef void build_dPu(const double[:, ::1] AL, const double[:, ::1] AR, const double[::1] expwt,
                    const double[::1] w, const double[::1] exptheta, npy_intp n, npy_intp u,
                    double t, const npy_intp[::1] inds, double[:, ::1] temp1,
                    double[:, ::1] temp2, double[:, ::1] dPu) nogil:

    cdef npy_intp i, j
    # write dKu into temp1
    # memset(&temp1[0, 0], 0, n*n * sizeof(double))
    # dK_dtheta(exptheta, n, u, temp1)

    # Gu = AL.T * dKu * AR
    dK_dtheta_A(exptheta, n, u, inds, AR, temp1)
    cdgemm_TN(AL, temp1, temp2)
    # Gu is in temp2

    # Vu matrix in temp1
    for i in range(n):
        for j in range(n):
            if i != j:
                temp1[i, j] = (expwt[i] - expwt[j]) / (w[i] - w[j]) * temp2[i, j]
            else:
                temp1[i, i] = t * expwt[i] * temp2[i, j]

    # dPu = AR * Vu * AL.T
    cdgemm_NN(AR, temp1, temp2)
    cdgemm_NT(temp2, AL, dPu)


def loglikelihood(double[::1] theta, double[:, ::1] counts, npy_intp n,
                  npy_intp[::1] inds=None, double t=1, npy_intp n_threads=1):
    """Log likelihood and gradient of the log likelihood of a continuous-time
    Markov model.

    Parameters
    ----------
    theta : array of shape = (n*(n-1)/2 + n) for dense parameterization, or shorter
        The free parameters of the model. See the section titled
        Parameterization.
    counts : array of shape = (n, n)
        The matrix of observed transition counts.
    n : int
        The size of `counts`
    inds : array of shape matching theta, or None
        For sparse parameterization, the active indices.
    t : double
        The lag time.
    n_threads : int
        The number of threads to use

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
    cdef int thread_num = 0
    cdef double logl
    cdef double[::1] w, expwt, grad, exptheta, grad_u
    cdef double[:, ::1] K, transmat, AL, AR, temp
    cdef double[:, :, ::1] dPu
    cdef double[:, :, :, ::1] temp1

    grad = zeros(size)
    exptheta = zeros(size)
    K = zeros((n, n))
    temp = zeros((n, n))
    temp1 = zeros((n_threads, 2, n, n))
    dPu = zeros((n_threads, n, n))
    grad_u = zeros((n_threads))

    for u in range(size):
        exptheta[u] = exp(theta[u])

    buildK(exptheta, n, inds, K)
    if not np.all(np.isfinite(K)):
        # these parameters don't seem good...
        # tell the optimizer to stear clear!
        return -np.inf, ascontiguousarray(grad)


    AL, AR, w, expwt = dP_dtheta_terms(K, n, t)
    transmat = np.dot(np.dot(AR, np.diag(expwt)), AL.T)

    with nogil, parallel(num_threads=n_threads):
        IF OPENMP:
            thread_num = openmp.omp_get_thread_num()
        for u in prange(size):
            # write dP / dtheta_u into dPu[thread_num]
            build_dPu(AL, AR, expwt, w, exptheta, n, u, t, inds,
                      temp1[thread_num, 0], temp1[thread_num, 1], dPu[thread_num])

            grad_u[thread_num] = 0
            for i in range(n):
                for j in range(n):
                    grad_u[thread_num] += counts[i, j] * (dPu[thread_num, i, j] / transmat[i, j])
            grad[u] = grad_u[thread_num]

    logl = 0
    for i in range(n):
        for j in range(n):
            logl += counts[i, j] * log(transmat[i, j])

    return logl, ascontiguousarray(grad)


def hessian(double[::1] theta, double[:, ::1] counts, npy_intp n, double t=1,
            npy_intp[::1] inds=None, npy_intp n_threads=1):
    cdef npy_intp size = (n*(n-1)/2) + n
    if not (counts.shape[0] == n and counts.shape[1] == n):
        raise ValueError('counts must be n x n')
    if not theta.shape[0] == size:
        raise ValueError('theta must have length (n*(n-1)/2) + n')

    cdef npy_intp u, v, i, j
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

    buildK(exptheta, n, None, K)
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

    return np.asarray(hessian)


def uncertainty_K(const double[:, :] invhessian, const double[::1] theta,
                  npy_intp n, npy_intp[::1] inds=None, npy_intp n_threads=1):
    cdef npy_intp size = (n*(n-1)/2) + n
    cdef npy_intp u, v, i, j
    cdef double[::1] exptheta
    cdef double[:, ::1] sigmaK, dKu, dKv, K, eye

    if not invhessian.shape[0] == size and invhessian.shape[1] == size:
        raise ValueError('counts must be n*(n-1)/2+n  x  n*(n-1)/2+n')
    if not theta.shape[0] == size:
        raise ValueError('theta must have length n*(n-1)/2+n')

    sigmaK = zeros((n, n))
    dKu = zeros((n, n))
    dKv = zeros((n, n))
    K = zeros((n, n))
    exptheta = np.exp(theta)
    eye = np.eye(n)

    buildK(exptheta, n, inds, K)

    for u in range(size):
        dK_dtheta_A(exptheta, n, u, inds, eye, dKu)
        for v in range(size):
            dK_dtheta_A(exptheta, n, v, inds, eye, dKv)
            # this could be optimized, since dKu and dKv are sparse and we
            # know their pattern
            for i in range(n):
                for j in range(n):
                    sigmaK[i,j] += invhessian[u,v] * dKu[i,j] * dKv[i,j]

    return np.asarray(sigmaK)


def uncertainty_pi(const double[:, :] invhessian, const double[::1] theta, npy_intp n):
    cdef npy_intp i
    cdef npy_intp size = (n*(n-1)/2) + n
    if not invhessian.shape[0] == size and invhessian.shape[1] == size:
        raise ValueError('counts must be n*(n-1)/2+n  x  n*(n-1)/2+n')
    if not theta.shape[0] == size:
        raise ValueError('theta must have length n*(n-1)/2+n')

    cdef double[::1] pi = np.exp(theta)[n*(n-1)/2:]

    sigma_pi = zeros(n)
    for i in range(n):
        sigma_pi[i] = pi[i] * invhessian[n*(n-1)/2 + i, n*(n-1)/2 + i]

    return np.asarray(sigma_pi)
