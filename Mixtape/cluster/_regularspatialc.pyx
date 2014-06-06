from libc.math cimport sqrt
import numpy as np
import scipy.sparse as sp
cimport numpy as np
cimport cython


cdef int INITIAL_CENTERS_BUFFER_SIZE = 100
cdef int CENTERS_BUFFER_GROWTH_MULTIPLE = 2

cdef extern:
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)


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
    X_squared_norms : 2d array

    """
    cdef size_t i
    cdef double dist2
    cdef int n_features = X.shape[1]

    while Xi < len(X):
        for i in range(n_centers):
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist2 = 0.0
            dist2 += ddot(n_features, &X[Xi, 0], 1, &centers[i, 0], 1)
            dist2 *= -2
            dist2 += centers_squared_norms[i]
            dist2 += x_squared_norms[Xi]

            if dist2 < d2_min:
                break
        else:
            return Xi
        Xi += 1
    return -1


def _rspatial_euclidean(double[:, ::1] X, double d_min):
    cdef Py_ssize_t i
    cdef size_t n_centers, realloc_length
    cdef size_t n_features = X.shape[1]
    cdef double d2_min = d_min*d_min
    cdef double[::1] x_squared_norms = np.sum(np.square(X), axis=1)
    cdef double[:, ::1] centers_buffer, centers_buffer_new
    cdef double[::1] centers_squared_norms_buffer, centers_squared_norms_buffer_new

    centers_buffer = np.zeros((INITIAL_CENTERS_BUFFER_SIZE, n_features))
    centers_squared_norms_buffer = np.zeros(INITIAL_CENTERS_BUFFER_SIZE)
    n_centers = 1
    centers_buffer[0] = X[0]
    centers_squared_norms_buffer[0] = x_squared_norms[0]

    i = 1
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
                centers_squared_norms_buffer, n_centers, i, d2_min)
        if i < 0:
            # i=-1 is a signal that the algorithim is over
            break
        centers_buffer[n_centers] = X[i]
        centers_squared_norms_buffer[n_centers] = x_squared_norms[i]
        n_centers += 1

    return np.asarray(centers_buffer[:n_centers])
