"""
Compute-intensive kernels for implementing reversible hidden Markov models
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2013, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import warnings
import scipy.optimize
cimport cython
cimport numpy as np
cdef extern from "math.h":
    double HUGE_VAL
    double exp(double)
    double log(double)

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------
DTYPE = np.float64
ctypedef np.float64_t DTYPE_T

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


def reversible_transmat(np.ndarray[ndim=2, dtype=DTYPE_T] counts):
    """Calculate the maximum likelihood transition probability matrix given
    observed transition counts at equilibrium.

    Parameters
    ----------
    counts : np.ndarray, shape=[n_states, n_states]
        `counts[i,j] holds the number of observed transitions from state i
        to state j.

    Returns
    -------
    transmat : np.ndarray, shape=[n_states, n_states]
        The maximum likelihood transition matrix that satisfies detailed
        balance (i.e. a single stationary eigenvector, pi that satisfies
        `dot T = pi T'`
    populations : np.ndarray, shape[n_states]
        The stationary eigenvector of the transition matrix
    
    Notes
    -----
    This method based on notes by Kyle A. Beauchamp on the reversible
    transition matix likelihood function included with the MSMBuilder
    distribution (docs/notes/mle_notes.pdf).
    """
    counts = np.asarray(counts, dtype=DTYPE)
    cdef int n_states = counts.shape[1]
    triu_indices = np.triu_indices(n_states)
    tril_indices = np.tril_indices(n_states)
    if counts.shape[0] != counts.shape[1]:
        raise TypeError('Counts must be a symmetric two-dimensional array')

    symcounts = (counts + counts.T - np.diag(np.diag(counts)))[triu_indices]
    rowsums = np.sum(counts, axis=1)
    logrowsums = np.log(rowsums)
    u0 = np.ones_like(symcounts)

    r = scipy.optimize.minimize(reversible_transmat_likelihood, u0, jac=reversible_transmat_grad,
                                args=(symcounts, rowsums, logrowsums),
                                method='l-bfgs-b')
    if not r['success']:
        warnings.warn('Maximum likelihood reversible transition matrix'
                      'optimization failed: %s' % r['message'])
    reversible_counts = np.zeros((n_states, n_states))
    reversible_counts[triu_indices] = r['x']
    reversible_counts[tril_indices] = r['x']

    equilibrium = reversible_counts.sum(axis=0) / reversible_counts.sum()
    transmat = reversible_counts / np.sum(reversible_counts, axis=1)[:, np.newaxis]
    return transmat, equilibrium


@cython.wraparound(False)
@cython.boundscheck(False)
def reversible_transmat_grad(
        np.ndarray[ndim=1, dtype=DTYPE_T] u not None,
        np.ndarray[ndim=1, dtype=DTYPE_T] symcounts not None,
        np.ndarray[ndim=1, dtype=DTYPE_T] rowsums not None,
        np.ndarray[ndim=1, dtype=DTYPE_T] logrowsums not None):
    """Calculate the gradient of the negative log-likelihood
    of a reversible transition matrix with respect to the indepenent
    variables.

    Parameters
    ----------
    u : np.array, ndim=1
        The free parameters. These are the log of the upper triangular
        transition matrix entries in symmetric storage.
    symcounts : np.ndarray, ndim=1,
        The number of observed transitions from i to j plus
        the number of observed counts from j to i, minus the number of
        counts from i to i (so that i->i *is not* double-counted in
        symcounts. e.g. `counts + counts.T - np.diag(np.diag(counts))`
    rowsums : np.ndarray, ndim=1
        The row sums of counts, `np.sum(counts, axis=1)`
    logrowsums : np.ndarray, ndim=1
        The natural log of the row sums
    """
    cdef int i, j, k
    cdef int n_states = len(rowsums)
    cdef int n_entries = (n_states*(n_states+1))//2
    assert len(u) == n_entries
    assert len(symcounts) == n_entries
    cdef np.ndarray[ndim=1, dtype=DTYPE_T] grad = np.empty_like(u)
    cdef np.ndarray[ndim=1, dtype=DTYPE_T] q = logsymsumexp(u, n_states)
    cdef np.ndarray[ndim=1, dtype=DTYPE_T] v = np.exp(logrowsums - q)

    k = 0
    for i in range(n_states):
        grad[k] = symcounts[k] - exp(u[k]) * v[i]
        k += 1
        for j in range(i+1, n_states):
            grad[k] = symcounts[k] - np.exp(u[k]) * (v[i] + v[j])
            k += 1
    return -grad


@cython.wraparound(False)
@cython.boundscheck(False)
def reversible_transmat_likelihood(
        np.ndarray[ndim=1, dtype=DTYPE_T] u not None,
        np.ndarray[ndim=1, dtype=DTYPE_T] symcounts not None,
        np.ndarray[ndim=1, dtype=DTYPE_T] rowsums not None,
        np.ndarray[ndim=1, dtype=DTYPE_T] logrowsums not None):
    """Calculate the negative log-likelihood of a reversible transition
    matrix given observed transition counts

    Parameters
    ----------
    u : np.array, ndim=1
        The free parameters. These are the log of the upper triangular
        transition matrix entries in symmetric storage.
    symcounts : np.ndarray, ndim=1,
        The number of observed transitions from i to j plus
        the number of observed counts from j to i, minus the number of
        counts from i to i (so that i->i *is not* double-counted in
        symcounts. e.g. `counts + counts.T - np.diag(np.diag(counts))`
    rowsums : np.ndarray, ndim=1
        The row sums of counts, `np.sum(counts, axis=1)`
    logrowsums : np.ndarray, ndim=1
        The natural log of the row sums
    """
    cdef int n_states = len(rowsums)
    likelihood = np.dot(u, symcounts) - np.dot(rowsums, logsymsumexp(u, n_states))
    return -likelihood


cdef inline DTYPE_T max(DTYPE_T a, DTYPE_T b):
    return a if a > b else b


@cython.wraparound(False)
@cython.boundscheck(False)
def logsymsumexp(np.ndarray[ndim=1, dtype=DTYPE_T] x not None,
                 int n):
    """Calculate the log-sum-exp of the rows (or columns) of a symmetric
    matrix stored in symmetric storage.
    
    Symmetric storage is a simple format for storing a 2d symmetric matrix
    in a 1 dimensional array of length N*(N+1)/2. For a 5x5 matrix, the
    indices in the 1-d vector for each upper triangular matrix entry are as
    shown below:

    [0  1  2   3  4]
    [-  5  6   7  8]
    [-  -  9  10 11]
    [-  -  -  12 13]
    [-  -  -  -  14]
    """
    cdef int i, j, k
    if n*(n+1)//2 != len(x):
        raise ValueError('Incompatible vector size. It must be a binomial '
                         'coefficient n choose 2 for some integer n >= 2.')
    log_sums = np.zeros(n, dtype=DTYPE)
    maxes = np.empty(n, dtype=DTYPE)
    maxes.fill(-HUGE_VAL)

    k = 0
    for i in range(n):
        for j in range(i, n):
            maxes[j] = max(x[k], maxes[j])
            maxes[i] = max(x[k], maxes[i])
            k += 1

    k = 0
    for i in range(n):
        log_sums[i] += exp(x[k] - maxes[i])
        k += 1
        for j in range(i+1, n):
            log_sums[i] += exp(x[k] - maxes[i])
            log_sums[j] += exp(x[k] - maxes[j])
            k += 1

    for i in range(n):
        log_sums[i] = log(log_sums[i]) + maxes[i]
    return log_sums
