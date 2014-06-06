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

cdef ddot_t *ddot = <ddot_t*>f2py_pointer(scipy.linalg.blas.ddot._cpointer)
cdef sdot_t *sdot = <sdot_t*>f2py_pointer(scipy.linalg.blas.sdot._cpointer)
cdef idamax_t *idamax = <idamax_t*>f2py_pointer(scipy.linalg.blas.idamax._cpointer)
cdef idamax_t *isamax = <idamax_t*>f2py_pointer(scipy.linalg.blas.idamax._cpointer)

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
        real[::1] x_squared_norms
        real[::1] cluster_center_squared_norms
        real[:, ::1] cluster_centers
        double[::1] distances = np.inf * ones(n_samples, dtype=np.double)
        np.int64_t[::1] labels = zeros(n_samples, dtype=np.int64)
        int one = 1

    if real == double:
        cluster_centers = zeros((n_clusters, n_features), dtype=np.double)
        cluster_center_squared_norms = zeros(n_clusters, dtype=np.double)
        x_squared_norms = zeros(n_samples, dtype=np.double)
        for j in range(n_samples):
            x_squared_norms[j] = ddot(&n_features, <double*> &X[j,0], &one,
                                      <double*> &X[j,0], &one)
    else:
        cluster_centers = zeros((n_clusters, n_features), dtype=np.float32)
        cluster_center_squared_norms = zeros(n_clusters, dtype=np.float32)
        x_squared_norms = zeros(n_samples, dtype=np.float32)
        for j in range(n_samples):
            x_squared_norms[j] = sdot(&n_features, <np.float32_t*> &X[j,0], &one,
                                      <np.float32_t*> &X[j,0], &one)


    for i in range(n_clusters):
        memcpy(&cluster_centers[i, 0], &X[seed, 0], n_features*sizeof(real))
        cluster_center_squared_norms[i] = x_squared_norms[seed]

        for j in range(n_samples):
            dist = cluster_center_squared_norms[i] + x_squared_norms[j]
            if real == double:
                dist += - 2*ddot(&n_features, <double*> &cluster_centers[i, 0], &one,
                                 <double*> &X[j, 0], &one)
            else:
                dist += - 2*sdot(&n_features, <np.float32_t*> &cluster_centers[i, 0], &one,
                                 <np.float32_t*> &X[j, 0], &one)

            if dist < distances[j]:
                distances[j] = dist
                labels[j] = i

        # -1 needed because of fortran 1-based indexing
        if real == double:
            seed = idamax(&n_samples, &distances[0], &one) - 1
        else:
            seed = isamax(&n_samples, &distances[0], &one) - 1

    for j in range(n_samples):
        distances[j] = sqrt(distances[j])
    
    return array(cluster_centers), array(distances), array(labels)
