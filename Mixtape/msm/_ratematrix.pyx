#cython: boundscheck=False, cdivision=True, wraparound=False
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

from __future__ import print_function
import numpy as np
from numpy import (zeros, allclose, array, real, ascontiguousarray, dot, diag)
import scipy.linalg
from scipy.linalg import blas, eig
from numpy cimport npy_intp
from libc.math cimport sqrt, log, exp
from libc.string cimport memset

include "cy_blas.pyx"


cpdef buildK(double[::1] exptheta, npy_intp n, double[:, ::1] out):
    assert out.shape[0] == n
    assert out.shape[1] == n
    assert exptheta.shape[0] == (n*(n-1)/2) + n
    cdef npy_intp i, j, u
    cdef double K_ij, K_ji, s_ij
    cdef npy_intp n_S_triu = (n*(n-1)/2)
    cdef double[::1] pi = exptheta[n_S_triu:]
    cdef double[::1] K_ii = zeros(n)

    u = 0
    for i in range(n):
        for j in range(i+1, n):
            s_ij = exptheta[u]
            K_ij = s_ij * sqrt(pi[i] / pi[j])
            K_ji = s_ij * sqrt(pi[j] / pi[i])
            out[i, j] = K_ij
            out[j, i] = K_ji
            K_ii[i] -= K_ij
            K_ii[j] -= K_ji
            u += 1

    for i in range(n):
        out[i, i] = K_ii[i]

    assert np.allclose(np.array(out).sum(axis=1), 0.0)
    assert np.allclose(scipy.linalg.expm(np.array(out)).sum(axis=1), 1)
    assert np.all(0 < scipy.linalg.expm(np.array(out)))
    assert np.all(1 > scipy.linalg.expm(np.array(out)))


cpdef dK_dtheta(double[::1] exptheta, npy_intp n, npy_intp u, double[:, ::1] out):
    # workspace of size (n)

    cdef npy_intp n_S_triu = (n*(n-1)/2)
    assert out.shape[0] == n
    assert out.shape[1] == n
    assert u >= 0 and u < n_S_triu + n
    assert exptheta.shape[0] == n_S_triu + n

    cdef npy_intp i, j
    cdef double dK_ij
    cdef double[::1] pi = exptheta[n_S_triu:]
    cdef double[::1] dK_ii

    if u < n_S_triu:
        # the perturbation is to the triu rate matrix
        # first, use the linear index, u, to get the (i,j)
        # indices of the symmetric rate matrix
        i = n - 2 - <int>(sqrt(-8*u + 4*n*(n-1)-7)/2.0 - 0.5)
        j = u + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2

        s_ij = exptheta[u]
        dK_ij = s_ij * sqrt(pi[i] / pi[j])
        dK_ji = s_ij * sqrt(pi[j] / pi[i])

        out[i, j] = dK_ij
        out[j, i] = dK_ji
        out[i, i] = -dK_ij
        out[j, j] = -dK_ji

    else:
        # the perturbation is to the equilibrium distribution

        # `i` is now the index, in `pi`, of the perturbed element
        # of the equilibrium distribution
        i = u - n_S_triu
        dK_ii = zeros(n)

        for j in range(n):
            if j == i:
                continue

            if j > i:
                k = (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
            else:
                k = (n*(n-1)/2) - (n-j)*((n-j)-1)/2 + i - j - 1

            s_ij = exptheta[k]
            dK_ij = 0.5  * s_ij * sqrt(pi[i] / pi[j])
            dK_ji = -0.5 * s_ij * sqrt(pi[j] / pi[i])

            out[i, j] = dK_ij
            out[j, i] = dK_ji
            dK_ii[i] -= dK_ij
            dK_ii[j] -= dK_ji

        for i in range(n):
            out[i, i] = dK_ii[i]


cpdef dP_dtheta_terms(double[:, ::1] K, npy_intp n):
    cdef npy_intp i
    w, AL, AR = scipy.linalg.eig(K, left=True, right=True)

    for i in range(n):
        # we need to ensure the proper normalization
        AL[:, i] /= dot(AL[:, i], AR[:, i])

    assert np.allclose(scipy.linalg.inv(AR).T, AL)

    AL = ascontiguousarray(real(AL))
    AR = ascontiguousarray(real(AR))
    w = ascontiguousarray(real(w))
    expw = zeros(w.shape[0])
    for i in range(w.shape[0]):
        expw[i] = exp(w[i])

    return AL, AR, w, expw


def loglikelihood(double[::1] theta, double[:, ::1] counts, npy_intp n):
    cdef npy_intp size = (n*(n-1)/2) + n
    if not (counts.shape[0] == n and counts.shape[1] == n):
        raise ValueError('counts must be n x n')
    if not theta.shape[0] == size:
        raise ValueError('theta must have length (n*(n-1)/2) + n')

    cdef npy_intp u, i, j
    cdef double grad_u, objective
    cdef double[::1] w, expw, grad, exptheta
    cdef double[:, ::1] AL, AR, Gu, transmat, temp, Vu, dPu, dKu, K

    grad = zeros(size)
    exptheta = zeros(size)
    temp = zeros((n, n))
    K = zeros((n, n))
    Vu = zeros((n, n))
    dKu = zeros((n, n))
    for i in range(size):
        exptheta[i] = exp(theta[i])

    buildK(exptheta, n, K)
    AL, AR, w, expw = dP_dtheta_terms(K, n)

    cdgemm_NN(AR, diag(expw), temp)
    transmat = K
    cdgemm_NT(temp, AL, transmat)
    # Equivalent to  transmat = AR * diag(expw) * AL.T
    # placing the results in the same memory as K (destroying K)
    assert np.allclose(transmat, np.dot(np.dot(AR,  np.diag(expw)), AL.T))

    for u in range(size):
        memset(&dKu[0, 0], 0, n*n * sizeof(double))
        dK_dtheta(exptheta, n, u, dKu)

        cdgemm_TN(AL, dKu, temp)
        Gu = dKu
        cdgemm_NN(temp, AR, Gu)
        # Equivalent to Gu = AL.T * dKu * AR
        # placing results in same memory as dKu (destroying dKu)

        for i in range(n):
            for j in range(n):
                if i != j:
                    Vu[i, j] = (expw[i] - expw[j]) / (w[i] - w[j]) * Gu[i, j]
                else:
                    Vu[i, i] = expw[i] * Gu[i, j]

        cdgemm_NN(AR, Vu, temp)
        dPu = Vu
        cdgemm_NT(temp, AL, dPu)
        # Equivalent to dPu = AR * Vu * AL.T
        # placing results in same memory as Vu (destroying Vu)

        grad_u = 0
        for i in range(n):
            for j in range(n):
                grad_u += counts[i, j] * (dPu[i, j] / transmat[i, j])
        grad[u] = grad_u

    objective = 0
    for i in range(n):
        for j in range(n):
            objective += counts[i, j] * log(transmat[i, j])

    return objective, ascontiguousarray(grad)
