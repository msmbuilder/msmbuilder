# cython: c_string_type=str, c_string_encoding=ascii

# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.

from __future__ import print_function
import numpy as np
import mdtraj as md
from libc.float cimport FLT_MAX
from libc.string cimport strcmp
from numpy cimport npy_intp

__all__ = ['assign_nearest', 'pdist', 'dist']

cdef VECTOR_METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev",
                       "canberra", "braycurtis", "hamming", "jaccard",
                       "cityblock")
cdef const char* RMSD = "rmsd"

#-----------------------------------------------------------------------------
# extern
#-----------------------------------------------------------------------------

cdef extern from "assign.hpp":
    double assign_nearest_double(const double* X, const double* Y,
        const char* metric, const npy_intp* X_indices, npy_intp n_X,
        npy_intp n_Y, npy_intp n_features, npy_intp n_X_indices,
        npy_intp* assignments) nogil
    double assign_nearest_float(const float* X, const float* Y,
        const char* metric, const npy_intp* X_indices, npy_intp n_X,
        npy_intp n_Y, npy_intp n_features, npy_intp n_X_indices,
        npy_intp* assignments) nogil
cdef extern from "pdist.hpp":
    void pdist_double(const double* X, const char* metric, npy_intp n, npy_intp m,
        double* out) nogil
    void pdist_float(const float* X, const char* metric, npy_intp n, npy_intp m,
        double* out) nogil
    void pdist_double_X_indices(const double* X, const char* metric, npy_intp n,
        npy_intp m, const npy_intp* X_indices, npy_intp n_X_indices,
        double* out) nogil
    void pdist_float_X_indices(const float* X, const char* metric, npy_intp n,
        npy_intp m, const npy_intp* X_indices, npy_intp n_X_indices,
        double* out) nogil
cdef extern from "cdist.hpp":
    void cdist_double(const double* XA, const double* XB, const char* metric,
        npy_intp na, npy_intp nb, npy_intp m, double* out) nogil
    void cdist_float(const float* XA, const float* XB, const char* metric,
        npy_intp na, npy_intp nb, npy_intp m, double* out) nogil
cdef extern from "dist.hpp":
    void dist_double(const double* X, const double* y, const char* metric,
        npy_intp n, npy_intp m, double* out) nogil
    void dist_float(const float* X, const float* y, const char* metric,
        npy_intp n, npy_intp m, double* out) nogil
    void dist_double_X_indices(const double* X, const double* y, const char* metric,
        npy_intp n, npy_intp m, const npy_intp* X_indices, npy_intp n_X_indices,
        double* out) nogil
    void dist_float_X_indices(const float* X, const float* y, const char* metric,
        npy_intp n, npy_intp m, const npy_intp* X_indices, npy_intp n_X_indices,
        double* out) nogil
cdef extern from "sumdist.hpp":
    double sumdist_double(const double* X, const char* metric, npy_intp n,
        npy_intp m, const npy_intp* pairs, npy_intp p) nogil
    double sumdist_float(const float* X, const char* metric, npy_intp n,
        npy_intp m, const npy_intp* pairs, npy_intp p) nogil
cdef extern from "center.h":
    void inplace_center_and_trace_atom_major(float* coords, float* traces,
        const int n_frames, const int n_atoms) nogil
cdef extern from "theobald_rmsd.h":
    cdef extern float msd_atom_major(int nrealatoms, int npaddedatoms, float* a,
        float* b, float G_a, float G_b, int computeRot, float rot[9]) nogil

cdef extern from "math.h":
    float sqrt(float x) nogil

#-----------------------------------------------------------------------------
# Public interface functions
#-----------------------------------------------------------------------------


def assign_nearest(X, Y, const char* metric, npy_intp[::1] X_indices=None):
    """assign_nearest(X, Y, metric, X_indices=None)

    For each point in X, compute the index of the nearest element in Y.

    Parameters
    ----------
    X : array, shape = (n_samples_X, n_features) or md.Trajectory
        The data array
    Y : array, shape = (n_samples_Y, n_features) or md.Trajectory
        The array of cluster centers
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that both X
        and cluster centers be of type md.Trajectory; other distance metrics
        require that they be arrays.
    X_indices : array of indices, or None
        If supplied, only data points with index in X_indices will be
        considered. `X_indices = None` is equivalent to
        `X_indices = range(len(X))`

    Returns
    -------
    assignments : array, shape=(len(X),), or shape=(len(X_indices),)
        For each point in `X`, or `X[X_indices]`, the index of the nearest
        point in `Y`.
    inertia : double
        The sum of the distance from each point in X[X_indices] to its assigned
        point in `Y`.

    See Also
    --------
    mdtraj.rmsd
    """
    if (isinstance(X, md.Trajectory) and isinstance(Y, md.Trajectory) and strcmp(metric, RMSD) == 0):
        return _assign_nearest_rmsd(X, Y, X_indices)


    if not isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        raise TypeError()
    if metric not in VECTOR_METRICS:
        raise ValueError('metric must be one of %s' %
                         ', '.join("'%s'" % s for s in VECTOR_METRICS))

    if X.dtype == np.float64 and Y.dtype == np.float64:
        return _assign_nearest_double(X, Y, metric, X_indices)
    elif X.dtype == np.float32 and Y.dtype == np.float32:
        return _assign_nearest_float(X, Y, metric, X_indices)
    else:
        raise TypeError('X and y must be both float32 or float64')


def cdist(XA, XB, const char* metric):
    """cdist(XA, XB, metric):

    Computes distance between each pair of the two collections of inputs.

    Parameters
    ----------
    XA : array, shape = (na_samples, m_features) or md.Trajectory
        The data array
    XB : array, shape = (nb_samples, m_features) or md.Trajectory
        The reference array
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that X be of type
        md.Trajectory; other distance metrics require that it be a numpy array

    Returns
    -------
    Y : array, shape = (na_samples, nb_samples)
        For each sample i in XA and each sample j in XB
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in Y[i,j]

    Note
    ----
    Not implemented for X_indices != None

    See Also
    --------
    mdtraj.rmsd
    scipy.spatial.distance.cdist
    """
    if (isinstance(XA, md.Trajectory) and isinstance(XB, md.Trajectory) and strcmp(metric, RMSD) == 0):
        return _cdist_rmsd(XA, XB)

    if not (isinstance(XA, np.ndarray) and isinstance(XB, np.ndarray)):
        raise TypeError('XA and XB must be numpy arrays')
    if metric not in VECTOR_METRICS:
        raise ValueError('metric must be one of %s' %
                         ', '.join("'%s'" % s for s in VECTOR_METRICS))

    if XA.dtype == np.float64 and XB.dtype == np.float64:
        return _cdist_double(XA, XB, metric)
    elif XA.dtype == np.float32 and XB.dtype == np.float32:
        return _cdist_float(XA, XB, metric)
    else:
        raise TypeError('XA and XB must be identically float32 or float64')


def pdist(X, const char* metric, npy_intp[::1] X_indices=None):
    """pdist(X, metric, X_indices=None)

    Pairwise distances between observations

    Parameters
    ----------
    X : array, shape = (n_samples, n_features) or md.Trajectory
        The data array
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that X be of type
        md.Trajectory; other distance metrics require that it be a numpy array
    X_indices : array of indices, or None
        If supplied, only data points with index in X_indices will be considered.
        `X_indices = None` is equivalent to `X_indices = range(len(X))`

    Returns
    -------
    dist : ndarray, size=(len(X) choose 2,) or shape=(len(X_indices) choose 2,)
        Returns a condensed distance matrix `dist`.  For
        each :math:`i` and :math:`j` (where :math:`i<j<n`), the
        metric ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``ij``.

    See Also
    --------
    mdtraj.rmsd
    scipy.spatial.distance.pdist
    scipy.spatial.distance.squareform
    """
    if (isinstance(X, md.Trajectory) and strcmp(metric, RMSD) == 0):
        return _pdist_rmsd(X, X_indices)

    if not isinstance(X, np.ndarray):
        raise TypeError()
    if metric not in VECTOR_METRICS:
        raise ValueError('metric must be one of %s' %
                         ', '.join("'%s'" % s for s in VECTOR_METRICS))

    if X.dtype == np.float64:
        return _pdist_double(X, metric, X_indices)
    elif X.dtype == np.float32:
        return _pdist_float(X, metric, X_indices)
    else:
        raise TypeError('X must be float32 or float64')


def dist(X, y, const char* metric, npy_intp[::1] X_indices=None):
    """dist(X, y, metric, X_indices=None)

    Distance from one point to many points.

    Parameters
    ----------
    X : array, shape = (n_samples, n_features) or md.Trajectory
        A data array
    y : array, shape = (n_features) or md.Trajectory of length 1
        A single data point
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that both X
        and cluster centers be of type md.Trajectory; other distance metrics
        require that they be arrays.

    Returns
    -------
    Y : ndarray, shape=(len(X),) or len(X_indices)
        The distance from Y to each `X` or `X[X_indices]`.

    See Also
    --------
    mdtraj.rmsd
    scipy.spatial.distance.cdist
    """
    if (isinstance(X, md.Trajectory) and isinstance(y, md.Trajectory) and strcmp(metric, RMSD) == 0):
        return _dist_rmsd(X, y, X_indices)

    if not isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        raise TypeError()
    if metric not in VECTOR_METRICS:
        raise ValueError('metric must be one of %s' %
                         ', '.join("'%s'" % s for s in VECTOR_METRICS))

    if X.dtype == np.float64 and y.dtype == np.float64:
        return _dist_double(X, y, metric, X_indices)
    elif X.dtype == np.float32 and y.dtype == np.float32:
        return _dist_float(X, y, metric, X_indices)
    else:
        raise TypeError('X and y must be both float32 or float64')


def sumdist(X, const char* metric, npy_intp[:, ::1] pair_indices):
    """sumdist(X, metric, pair_indices)

    Sum of the distance between pairs of points

    Parameters
    ----------
    X : array, shape = (n_samples, n_features) or md.Trajectory
        A data array
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that both X
        and cluster centers be of type md.Trajectory; other distance metrics
        require that they be arrays.
    pair_indices : array, shape = (n_pairs, 2)
        Each element in pair_indices is a tuple of two indices -- a pair of
        elements in X to include in the summation.

    Returns
    -------
    s : float
        The sum of the distance between each pair of elements:
        ``sum(dist(X[p[0]], X[p[1]]) for p in pair_indices)``
    """
    if (isinstance(X, md.Trajectory) and strcmp(metric, RMSD) == 0):
        return _sumdist_rmsd(X, pair_indices)

    if metric not in VECTOR_METRICS:
        raise ValueError('metric must be one of %s' %
                         ', '.join("'%s'" % s for s in VECTOR_METRICS))

    if X.dtype == np.float64:
        return _sumdist_double(X, metric, pair_indices)
    elif X.dtype == np.float32:
        return _sumdist_float(X, metric, pair_indices)
    else:
        raise TypeError('X must be both float32 or float64')


#-----------------------------------------------------------------------------
# Private implementation
#-----------------------------------------------------------------------------

cdef _assign_nearest_rmsd(X, Y, npy_intp[::1] X_indices=None):
    cdef npy_intp i, j
    assert (X.xyz.ndim == 3) and (Y.xyz.ndim == 3) and \
           (X.xyz.shape[2]) == 3 and (Y.xyz.shape[2] == 3)
    if not (X.xyz.shape[1]  == Y.xyz.shape[1]):
        raise ValueError("Input trajectories must have same number of atoms. "
                         "found %d and %d." % (X.xyz.shape[1], Y.xyz.shape[1]))

    cdef double inertia = 0
    cdef float min_d, rmsd
    cdef float[:, :, ::1] X_xyz = X.xyz
    cdef float[:, :, ::1] Y_xyz = Y.xyz
    cdef int n_atoms = X.xyz.shape[1]
    cdef npy_intp length
    cdef npy_intp X_length = X_xyz.shape[0]
    cdef npy_intp Y_length = Y_xyz.shape[0]
    cdef npy_intp[::1] assignments
    cdef float[::1] X_trace
    cdef float[::1] Y_trace

    if X._rmsd_traces is None:
        X.center_coordinates()
    if Y._rmsd_traces is None:
        Y.center_coordinates()
    X_trace = X._rmsd_traces
    Y_trace = Y._rmsd_traces

    if X_indices is None:
        length = X_length
        assignments = np.zeros(length, dtype=np.intp)

        for i in range(length):
            min_d = FLT_MAX;
            for j in range(Y_length):
                rmsd = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[i, 0, 0],
                    &Y_xyz[j, 0, 0], X_trace[i], Y_trace[j], 0, NULL))
                if rmsd < min_d:
                    min_d = rmsd;
                    assignments[i] = j;
            inertia += min_d;
    else:
        length = X_indices.shape[0]
        assignments = np.zeros(length, dtype=np.intp)

        for i in range(length):
            min_d = FLT_MAX;
            for j in range(Y_length):
                rmsd = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[X_indices[i], 0, 0],
                            &Y_xyz[j, 0, 0], X_trace[X_indices[i]], Y_trace[j], 0, NULL))
                if rmsd < min_d:
                    min_d = rmsd;
                    assignments[i] = j;
            inertia += min_d;

    return np.array(assignments, copy=False), inertia


cdef _assign_nearest_double(double[:, ::1] X, double[:, ::1] Y,
                            const char* metric, npy_intp[::1] X_indices=None):
    cdef npy_intp[::1] assignments
    cdef npy_intp length, n_features
    n_features = X.shape[1]
    assert n_features == Y.shape[1]
    if X_indices is None:
        length = X.shape[0]
    else:
        length = X_indices.shape[0]
    assignments = np.zeros(length, dtype=np.intp)

    cdef double inertia = assign_nearest_double(
        &X[0, 0], &Y[0, 0], metric,
        <npy_intp*> NULL if X_indices is None else &X_indices[0],
        X.shape[0], Y.shape[0], n_features, length,
        &assignments[0])
    return np.array(assignments, copy=False), inertia


cdef _assign_nearest_float(float[:, ::1] X, float[:, ::1] Y,
                           const char* metric, npy_intp[::1] X_indices=None):
    cdef npy_intp[::1] assignments
    cdef npy_intp length, n_features
    n_features = X.shape[1]
    assert n_features == Y.shape[1]
    if X_indices is None:
        length = X.shape[0]
    else:
        length = X_indices.shape[0]
    assignments = np.zeros(length, dtype=np.intp)

    cdef double inertia = assign_nearest_float(
        &X[0, 0], &Y[0, 0], metric,
        <npy_intp*> NULL if X_indices is None else &X_indices[0],
        X.shape[0], Y.shape[0], n_features, length,
        &assignments[0])
    return np.array(assignments, copy=False), inertia


cdef _cdist_rmsd(XA, XB):
    cdef npy_intp i, j
    cdef float[:, :, ::1] XA_xyz = XA.xyz
    cdef float[:, :, ::1] XB_xyz = XB.xyz
    cdef npy_intp XA_length = XA_xyz.shape[0]
    cdef npy_intp XB_length = XB_xyz.shape[0]
    cdef int n_atoms = XA_xyz.shape[1]
    cdef float[::1] XA_trace, XB_trace
    cdef double[:, ::1] out

    if XA._rmsd_traces is None:
        XA.center_coordinates()
    if XB._rmsd_traces is None:
        XB.center_coordinates()

    if XA_xyz.shape[1] != XB_xyz.shape[1]:
        raise ValueError('XA and XB must have the same number of atoms')

    XA_trace = XA._rmsd_traces
    XB_trace = XB._rmsd_traces

    out = np.zeros((XA_length, XB_length), dtype=np.double)

    for i in range(XA_length):
        for j in range(XB_length):
            rmsd = sqrt(msd_atom_major(n_atoms, n_atoms, &XA_xyz[i,0,0],
                        &XB_xyz[j,0,0], XA_trace[i], XB_trace[j], 0, NULL))
            out[i,j] = rmsd

    return np.array(out, copy=False)


cdef _cdist_double(double[:, ::1] XA, double[:, ::1] XB, const char* metric):
    cdef double[:, ::1] out
    if XA.shape[1] != XB.shape[1]:
        raise ValueError('XA and XB must have the same number of columns')
    out = np.zeros((XA.shape[0], XB.shape[0]), dtype=np.double)
    cdist_double(&XA[0,0], &XB[0,0], metric, XA.shape[0], XB.shape[0],
                    XA.shape[1], &out[0,0])
    return np.array(out, copy=False)

cdef _cdist_float(float[:, ::1] XA, float[:, ::1] XB, const char* metric):
    cdef double[:, ::1] out
    if XA.shape[1] != XB.shape[1]:
        raise ValueError('XA and XB must have the same number of columns')
    out = np.zeros((XA.shape[0], XB.shape[0]), dtype=np.double)
    cdist_float(&XA[0,0], &XB[0,0], metric, XA.shape[0], XB.shape[0],
                    XA.shape[1], &out[0,0])
    return np.array(out, copy=False)


cdef _pdist_rmsd(X, npy_intp[::1] X_indices=None):
    cdef npy_intp i, j, k

    cdef double[::1] out
    cdef float[:, :, ::1] X_xyz = X.xyz
    cdef int n_atoms = X.xyz.shape[1]
    cdef npy_intp X_length = X_xyz.shape[0]
    cdef float[::1] X_trace

    if X._rmsd_traces is None:
        X.center_coordinates()
    X_trace = X._rmsd_traces

    if X_indices is None:
        out = np.zeros(X_xyz.shape[0] * (X_xyz.shape[0] - 1) / 2, dtype=np.double)

        k = 0
        for i in range(X_xyz.shape[0]):
            for j in range(i+1, X_xyz.shape[0]):
                rmsd = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[i, 0, 0],
                            &X_xyz[j, 0, 0], X_trace[i], X_trace[j], 0, NULL))
                out[k] = rmsd
                k += 1
    else:
        out = np.zeros(X_indices.shape[0] * (X_indices.shape[0] - 1) / 2, dtype=np.double)

        k = 0
        for i in range(X_indices.shape[0]):
            for j in range(i+1, X_indices.shape[0]):
                rmsd = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[X_indices[i], 0, 0],
                            &X_xyz[X_indices[j], 0, 0], X_trace[X_indices[i]],
                            X_trace[X_indices[j]], 0, NULL))
                out[k] = rmsd
                k += 1

    return np.array(out, copy=False)


cdef _pdist_double(double[:, ::1] X, const char* metric, npy_intp[::1] X_indices=None):
    cdef double[::1] out
    if X_indices is None:
        out = np.zeros(X.shape[0] * (X.shape[0] - 1) / 2, dtype=np.double)
        pdist_double(&X[0,0], metric, X.shape[0], X.shape[1], &out[0])
    else:
        out = np.zeros(X_indices.shape[0] * (X_indices.shape[0] - 1) / 2, dtype=np.double)
        pdist_double_X_indices(&X[0, 0], metric, X.shape[0], X.shape[1],
            &X_indices[0], X_indices.shape[0], &out[0])

    return np.array(out, copy=False)


cdef _pdist_float(float[:, ::1] X, const char* metric, npy_intp[::1] X_indices=None):
    cdef double[::1] out
    if X_indices is None:
        out = np.zeros(X.shape[0] * (X.shape[0] - 1) / 2, dtype=np.double)
        pdist_float(&X[0,0], metric, X.shape[0], X.shape[1], &out[0])
    else:
        out = np.zeros(X_indices.shape[0] * (X_indices.shape[0] - 1) / 2, dtype=np.double)
        pdist_float_X_indices(&X[0, 0], metric, X.shape[0], X.shape[1],
            &X_indices[0], X_indices.shape[0], &out[0])
    return np.array(out, copy=False)


cdef _dist_rmsd(X, y, npy_intp[::1] X_indices=None):
    cdef npy_intp i, ii, j
    assert (X.xyz.ndim == 3) and (y.xyz.ndim == 3) and \
           (X.xyz.shape[2]) == 3 and (y.xyz.shape[2] == 3)
    if not (X.xyz.shape[1]  == y.xyz.shape[1]):
        raise ValueError("Input trajectories must have same number of atoms. "
                         "found %d and %d." % (X.xyz.shape[1], y.xyz.shape[1]))

    cdef double[::1] out
    cdef float[:, :, ::1] X_xyz = X.xyz
    cdef float[:, :, ::1] Y_xyz = y.xyz
    cdef int n_atoms = X.xyz.shape[1]
    cdef npy_intp X_length = X_xyz.shape[0]
    cdef npy_intp y_length = Y_xyz.shape[0]
    cdef float rmsd
    cdef float[::1] X_trace
    cdef float[::1] y_trace

    if X._rmsd_traces is None:
        X.center_coordinates()
    if y._rmsd_traces is None:
        y.center_coordinates()
    X_trace = X._rmsd_traces
    y_trace = y._rmsd_traces

    if X_indices is None:
        out = np.zeros(X_xyz.shape[0], dtype=np.double)
        for i in range(X_xyz.shape[0]):
            out[i] = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[i, 0, 0],
                          &Y_xyz[0, 0, 0], X_trace[i], y_trace[0], 0, NULL))
    else:
        out = np.zeros(X_indices.shape[0], dtype=np.double)
        for i in range(X_indices.shape[0]):
            out[i] = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[X_indices[i], 0, 0],
                          &Y_xyz[0, 0, 0], X_trace[X_indices[i]], y_trace[0], 0, NULL))
    return np.array(out, copy=False)


cdef _dist_double(double[:, ::1] X, double[::1] y, const char* metric, npy_intp[::1] X_indices=None):
    cdef double[::1] out
    assert X.shape[1] == y.shape[0]
    if X_indices is None:
        out = np.zeros(X.shape[0], dtype=np.double)
        dist_double(&X[0,0], &y[0], metric, X.shape[0], X.shape[1], &out[0])
    else:
        out = np.zeros(X_indices.shape[0], dtype=np.double)
        dist_double_X_indices(&X[0, 0], &y[0], metric, X.shape[0], X.shape[1],
            &X_indices[0], X_indices.shape[0], &out[0])
    return np.array(out, copy=False)


cdef _dist_float(float[:, ::1] X, float[::1] y, const char* metric, npy_intp[::1] X_indices=None):
    cdef double[::1] out
    assert X.shape[1] == y.shape[0]
    if X_indices is None:
        out = np.zeros(X.shape[0], dtype=np.double)
        dist_float(&X[0,0], &y[0], metric, X.shape[0], X.shape[1], &out[0])
    else:
        out = np.zeros(X_indices.shape[0], dtype=np.double)
        dist_float_X_indices(&X[0, 0], &y[0], metric, X.shape[0], X.shape[1],
            &X_indices[0], X_indices.shape[0], &out[0])
    return np.array(out, copy=False)


cdef double _sumdist_rmsd(X, npy_intp[:, ::1] pair_indices):
    if not pair_indices.shape[1] == 2:
        raise ValueError('pair_indices must be of shape = (n_pairs, 2)')

    cdef npy_intp i, ii, jj
    cdef double s = 0
    cdef float[:, :, ::1] X_xyz = X.xyz
    cdef int n_atoms = X.xyz.shape[1]
    cdef float[::1] X_trace

    if X._rmsd_traces is None:
        X.center_coordinates()
    X_trace = X._rmsd_traces

    for i in range(pair_indices.shape[0]):
        ii = pair_indices[i, 0]
        jj = pair_indices[i, 1]
        rmsd = sqrt(msd_atom_major(n_atoms, n_atoms, &X_xyz[ii, 0, 0],
                    &X_xyz[jj, 0, 0], X_trace[ii], X_trace[jj], 0, NULL))
        s += rmsd
    return s


cdef double _sumdist_double(double[:, ::1] X, const char* metric, npy_intp[:, ::1] pair_indices):
    if not pair_indices.shape[1] == 2:
        raise ValueError('pair_indices must be of shape = (n_pairs, 2)')
    return sumdist_double(&X[0,0], metric, X.shape[0], X.shape[1],
                          &pair_indices[0,0], pair_indices.shape[0])


cdef double _sumdist_float(float[:, ::1] X, const char* metric, npy_intp[:, ::1] pair_indices):
    if not pair_indices.shape[1] == 2:
        raise ValueError('pair_indices must be of shape = (n_pairs, 2)')
    return sumdist_float(&X[0,0], metric, X.shape[0], X.shape[1],
                          &pair_indices[0,0], pair_indices.shape[0])

