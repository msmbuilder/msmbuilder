# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import
import numpy as np
from sklearn.utils import check_random_state

from numpy cimport npy_intp
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from cpython.ref cimport PyObject

cdef extern from "src/kmedoids.h":
    void _kmedoids "kmedoids" (npy_intp nclusters, npy_intp nelements,
        double* distmatrix, npy_intp npass, npy_intp clusterid[],
        PyObject* random, double* error, npy_intp* ifound)
    map[npy_intp, npy_intp] _contigify_ids "contigify_ids" (
        npy_intp* ids, npy_intp length)


def kmedoids(npy_intp n_clusters, double[::1] distmatrix, npy_intp n_pass,
             npy_intp[::1] clusterid=None, random_state=None):
    """KMedoids clustering

    Arguments
    ---------
    n_clusters : int
        The number of clusters to be found.

    distmatrix : double array
        A condensed distance matrix of the pairwise distance between elements.
        This distance matrix should have the form of matrices produced by
        ``scipy.spatial.distance.pdist()``, or ``msmbuilder.libdistance.pdist``
        which is the lower triangular portion of a symmetric matrix in packed
        storage.

    n_pass : int
        The number of times clustering is performed. Clustering is performed
        n_pass times, each time starting from a different (random) initial
        assignment. The clustering solution with the lowest within-cluster
        sum of distances is chosen. If npass==0, then the clustering algorithm
        will be run once, where the initial assignment of elements to clusters
        is taken from the clusterid array.

    clusterid : int[nelements]
        On input, if npass==0, then clusterid contains the initial clustering
        assignment from which the clustering algorithm starts; all numbers in
        clusterid should be between zero and nelements-1 inclusive. If
        npass!=0, clusterid is ignored on input.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Returns
    --------
    clusterid : int[nelements]
        On output, clusterid contains the clustering solution that was found:
        clusterid contains the number of the cluster to which each item was
        assigned. On output, the number of a cluster is defined as the item
        number of the centroid of the cluster.

    error : double
      The sum of distances to the cluster center of each item in the optimal
      k-medoids clustering solution that was found.

    ifound : int
        If kmedoids is successful: the number of times the optimal clustering
        solution was found. The value of ifound is at least 1; its maximum
        value is npass. If the user requested more clusters than elements
        available, ifound is set to 0.
    """
    cdef double error
    cdef npy_intp i, ifound
    cdef npy_intp[::1] clusterid_
    cdef npy_intp n_elements
    n_elements = int(1 + np.sqrt(8*len(distmatrix) + 1) / 2.0)
    if len(distmatrix) != (n_elements * (n_elements-1) / 2):
        raise ValueError('len(distmatrix)=%s is not a valid size '
                         'of a condensed distance matrix, which should be of '
                         'size (N*(N-1)/2) for some positive integer, N' %
                         len(distmatrix))
    if n_clusters > n_elements:
        raise ValueError('Number of clusters requested (%d) greater than '
                         'number of elements (%d)' % (n_clusters, n_elements))
    if clusterid is not None and len(clusterid) != n_elements:
        raise ValueError('clusterid must be None or an array of length '
                         'n_elements')
    if n_pass < 0:
        raise ValueError('n_pass must be greater than or equal to zero.')

    # this is going to be the output, so we make a copy
    if clusterid is None:
        clusterid_ = np.zeros(n_elements, dtype=np.intp)
    else:
        clusterid_ = np.array(clusterid, dtype=np.intp, copy=True)

    random = check_random_state(random_state)

    # Call the underlying library
    _kmedoids(n_clusters, n_elements, &distmatrix[0], n_pass,
              &clusterid_[0], <PyObject*> random, &error, &ifound)

    return np.array(clusterid_, copy=False), error, ifound


def contigify_ids(npy_intp[::1] clusterids):
    """Renumber clusters to go from 0 to n_clusters - 1

    This function modifies the array inplace
    """
    cdef map[npy_intp, npy_intp] mapping
    mapping = _contigify_ids(&clusterids[0], clusterids.shape[0])
    return np.array(clusterids, copy=False), mapping
