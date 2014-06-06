#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from blas cimport *
from libc.math cimport sqrt
cimport cython
cimport numpy as np
import numpy as np
from numpy import zeros
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

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.float64_t _assign_labels_array(real[:, ::1] X,
                                        real[:, ::1] centers,
                                        long long[::1] labels,
                                        real[::1] distances):
    """Compute label assignment and inertia

    Parameters
    ----------
    X : 2d array [INPUT]
        The data array
    centers : 2d array [INPUT]
        The array with the cluster centers
    labels : 1d array [OUTPUT]
        The output labels for each data point (integers from 0 to n_clusters-1) will
        be written into this array
    distances : 1d array [OUTPUT]
        If distances is supplied and is an array of length == len(X), the distances
        from each X to its assigned cluster center will be written here

    Returns
    --------
    intertia : double
        The sum of the squares of the distance from each point to its assigned
        center
    """
    cdef:
        int n_clusters = centers.shape[0]
        int n_features = centers.shape[1]
        np.int64_t n_samples = X.shape[0]
        np.int64_t i, j
        int store_distances = 0
        np.float64_t inertia = 0.0
        real min_dist
        real dist
        real[::1] x_squared_norms
        real[::1] center_squared_norms
        int one = 1

    if n_samples == distances.shape[0]:
        store_distances = 1

    # First get the squared norms
    if real == double:
        center_squared_norms = zeros(n_clusters, dtype=np.double)
        x_squared_norms = zeros(n_samples, dtype=np.double)
        for j in range(n_samples):
            x_squared_norms[j] = ddot(&n_features, <double*> &X[j,0], &one,
                                      <double*> &X[j,0], &one)
        for j in range(n_clusters):
            center_squared_norms[j] = ddot(&n_features, <double*> &centers[j,0], &one,
                                           <double*> &centers[j,0], &one)
    else:
        center_squared_norms = zeros(n_clusters, dtype=np.float32)
        x_squared_norms = zeros(n_samples, dtype=np.float32)
        for j in range(n_samples):
            x_squared_norms[j] = sdot(&n_features, <np.float32_t*> &X[j,0], &one,
                                      <np.float32_t*> &X[j,0], &one)
        for j in range(n_clusters):
            center_squared_norms[j] = sdot(&n_features, <np.float32_t*> &centers[j,0], &one,
                                           <np.float32_t*> &centers[j,0], &one)


    for i in range(n_samples):
        min_dist = -1
        for j in range(n_clusters):
            # hardcoded: minimize euclidean distance to cluster center:
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist = x_squared_norms[i] + center_squared_norms[j]
            if real == double:
                dist += - 2*ddot(&n_features, <double*> &centers[j, 0], &one,
                                 <double*> &X[i, 0], &one)
            else:
                dist += - 2*sdot(&n_features, <np.float32_t*> &centers[j, 0], &one,
                                 <np.float32_t*> &X[i, 0], &one)

            if min_dist == -1 or dist < min_dist:
                labels[i] = j
                min_dist = dist

        if store_distances:
            distances[i] = sqrt(min_dist)
        inertia += min_dist

    return inertia
