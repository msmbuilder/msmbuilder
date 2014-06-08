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

from blas cimport *
from libc.math cimport sqrt
cimport cython
cimport numpy as np
from libc.string cimport memcpy
import numpy as np
from numpy import zeros, ones, array
import scipy.linalg.blas

__all__ = ['_kcenters_euclidean']

#-----------------------------------------------------------------------------
# Typedefs
#-----------------------------------------------------------------------------

ctypedef cython.floating real
cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL
cdef extern from "sdot.h":
    float sdot_(const int N, const float* x, const float* y)
cdef extern from "math.h":
    double HUGE_VAL
    float HUGE_VALF

cdef ddot_t *ddot = <ddot_t*>f2py_pointer(scipy.linalg.blas.ddot._cpointer)
cdef idamax_t *idamax = <idamax_t*>f2py_pointer(scipy.linalg.blas.idamax._cpointer)
cdef isamax_t *isamax = <isamax_t*>f2py_pointer(scipy.linalg.blas.isamax._cpointer)

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _kcenters_euclidean(real[:, ::1] X,
                          np.int64_t n_clusters,
                          np.int64_t seed):
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int new_center
        np.int64_t i, j, k
        real dist
        real[::1] distances
        real[::1] x_squared_norms
        real[::1] cluster_center_squared_norms
        real[:, ::1] cluster_centers
        np.int64_t[::1] labels = zeros(n_samples, dtype=np.int64)
        int one = 1

    if real == double:
        distances = zeros(n_samples, dtype=np.double)
        cluster_centers = zeros((n_clusters, n_features), dtype=np.double)
        cluster_center_squared_norms = zeros(n_clusters, dtype=np.double)
        x_squared_norms = zeros(n_samples, dtype=np.double)
        for j in range(n_samples):
            distances[j] = HUGE_VAL
            x_squared_norms[j] = ddot(&n_features, <double*> &X[j,0], &one,
                                      <double*> &X[j,0], &one)
    else:
        distances = zeros(n_samples, dtype=np.float32)
        cluster_centers = zeros((n_clusters, n_features), dtype=np.float32)
        cluster_center_squared_norms = zeros(n_clusters, dtype=np.float32)
        x_squared_norms = zeros(n_samples, dtype=np.float32)
        for j in range(n_samples):
            distances[j] = HUGE_VALF
            x_squared_norms[j] = sdot_(n_features, <float*> &X[j,0], <float*> &X[j,0])

    for i in range(n_clusters):
        memcpy(&cluster_centers[i, 0], &X[seed, 0], n_features*sizeof(real))
        cluster_center_squared_norms[i] = x_squared_norms[seed]

        for j in range(n_samples):
            dist = cluster_center_squared_norms[i] + x_squared_norms[j]
            if real == double:
                dist += - 2*ddot(&n_features, <double*> &cluster_centers[i, 0],
                                 &one, <double*> &X[j, 0], &one)
            else:
                dist += - 2*sdot_(n_features, <float*> &cluster_centers[i, 0],
                                  <float*> &X[j, 0])

            if dist < distances[j]:
                distances[j] = dist
                labels[j] = i

        # -1 needed because of fortran 1-based indexing
        if real == double:
            seed = idamax(&n_samples, <double*> &distances[0], &one) - 1
        else:
            seed = isamax(&n_samples, <float*> &distances[0], &one) - 1

    for j in range(n_samples):
        distances[j] = sqrt(distances[j])
    
    return array(cluster_centers), array(distances), array(labels)
