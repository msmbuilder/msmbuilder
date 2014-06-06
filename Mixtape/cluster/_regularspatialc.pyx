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

import numpy as np
cimport cython

cdef int INITIAL_CENTERS_BUFFER_SIZE = 8
cdef int CENTERS_BUFFER_GROWTH_MULTIPLE = 2

cdef extern:
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Py_ssize_t _rspatial_euclidean_next(
        double[:, ::1] X, double[::1] x_squared_norms,
        double[:, ::1] centers, double[::1] centers_squared_norms,
        size_t n_centers, size_t Xi, double d2_min):
    """Find the index of the *next* cluster center in regular spacial clustering
    with a euclidean distance metric

    Parameters
    ----------
    X : 2d array
        The data array
    x_squared_norms : 1d array
        Row sums of X squared. These can be calculated with just
        np.sum(X**2, axis=1)
    centers : 2d array
        The coordinates of the current centers
    centers_squared_norms : 1d array
        Row sums of the square of the centers
    n_centers : int
        The only portions of the centers array that we look at is
        centers[0:n_centers]. This is basically a memory optimization, because
        it means that the actual buffer holding the centers can be bigger, and
        you can just add a new center to the end without reallocing by just
        putting the data in and incrementing n_centers by one.
    Xi : int
        The index in X that we start looking from
    d2_min : double
        The square of the distance cutoff

    Returns
    -------
    newXi : int
        The index of the new cluster center

    Notes
    -----
    The pseudocode for this function is

        for i from Xi to len(X) - 1:
            if the squared distance from X[i] to each of the centers (from 0 \
                    to n_centers-1) is greater than d2_min

                return i to be the new centers
    """
    cdef size_t i
    cdef double dist2

    while Xi < X.shape[0]:
        for i in range(n_centers):
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist2 = -2*ddot(X.shape[1], &X[Xi, 0], 1, &centers[i, 0], 1) \
                + centers_squared_norms[i] \
                + x_squared_norms[Xi]

            if dist2 < d2_min:
                break
        else:
            return Xi
        Xi += 1
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _rspatial_euclidean(double[:, ::1] X, double d_min):
    """Regular spatial clustering with a euclidean distance metric

    Parameters
    ----------
    X : 2d array
        The data array
    d_min : double
        The minimum distance between clusters

    Returns
    -------
    cluster_centers : 2d array
        The return value is a subset of the data points in X that are all
        at least d_min apart from one another.
    """

    cdef Py_ssize_t i
    cdef size_t n_centers, realloc_length
    cdef size_t n_features = X.shape[1]
    cdef double d2_min = d_min*d_min
    cdef double[::1] x_squared_norms = np.einsum('ij,ij->i', X, X)
    cdef double[:, ::1] centers_buffer, centers_buffer_new
    cdef double[::1] centers_squared_norms_buffer, centers_squared_norms_buffer_new

    centers_buffer = np.zeros((INITIAL_CENTERS_BUFFER_SIZE, n_features))
    centers_squared_norms_buffer = np.zeros(INITIAL_CENTERS_BUFFER_SIZE)
    n_centers = 1
    centers_buffer[0] = X[0]
    centers_squared_norms_buffer[0] = x_squared_norms[0]

    i = 0
    centers_buffer_end = 1
    while i < X.shape[0]:
        # enlarge buffer if necessary
        if n_centers == centers_buffer.shape[0]:
            realloc_length = centers_buffer.shape[0]*CENTERS_BUFFER_GROWTH_MULTIPLE
            centers_buffer_new = np.zeros((realloc_length, n_features))
            centers_squared_norms_buffer_new = np.zeros(realloc_length)
            # copy over data
            centers_buffer_new[:n_centers, :] = centers_buffer[:, :]
            centers_squared_norms_buffer_new[:n_centers] = centers_squared_norms_buffer[:]
            # rename
            centers_buffer = centers_buffer_new
            centers_squared_norms_buffer = centers_squared_norms_buffer_new

        # advance through X to find the next cluster center
        i = _rspatial_euclidean_next(
                X, x_squared_norms, centers_buffer,
                centers_squared_norms_buffer, n_centers, i+1, d2_min)
        if i < 0:
            # i=-1 is a signal that the algorithim is over
            break
        centers_buffer[n_centers] = X[i]
        centers_squared_norms_buffer[n_centers] = x_squared_norms[i]
        n_centers += 1

    return np.asarray(centers_buffer[:n_centers])
