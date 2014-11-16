from __future__ import print_function, division, absolute_import
import numpy as np
from libc.stdlib cimport malloc, free


cdef extern from "src/cluster.h":
  void _kmedoids "kmedoids" (int nclusters, int nelements, double** distmatrix,
                            int npass, int clusterid[], double* error,
                            int* ifound)
cdef extern from "src/utils.h":
    int _triu_to_tril "triu_to_tril"(double* upper, size_t length,
                                     double* lower)
    int _contigify_ids "contigify_ids" (int* ids, size_t length)



def kmedoids(int n_clusters, double[::1] condensed_distmatrix, int n_pass,
             int[::1] clusterid=None):
    """KMedoids clustering

    Arguments
    ---------
    n_clusters : int
        The number of clusters to be found.

    condensed_distmatrix : double array
        A condensed distance matrix of the pairwise distance between elements.
        This distance matrix should have the form of matrices produced by
        scipy.spatial.distance.pdist(), which is basically the upper triangular
        portion of a symmetric matrix in packed storage:

        For example, with a 4x4 distance matrix, the upper triangular portion
        is:

            [x   d(0,1)  d(0,2)  d(0,3)]
            [x     x     d(1,2)  d(1,3)]
            [x     x        x    d(2,3)]
            [x     x        x       x  ]

        In condensed_distmatrix storage, this is laid out as a 1D array like:

            [d(0,1)  d(0,2)  d(0,3), d(1,2)  d(1,3), d(2,3)]

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
    cdef int i, ifound
    cdef int[::1] clusterid_
    cdef int n_elements = int(1 + np.sqrt(8*len(condensed_distmatrix) + 1)) / 2.0;
    if len(condensed_distmatrix) != (n_elements * (n_elements-1) / 2):
        raise ValueError('len(condensed_distmatrix)=%s is not a valid size '
                         'of a condensed distance matrix, which should be of '
                         'size (N*(N-1)/2) for some positive integer, N' %
                         len(condensed_distmatrix))
    if n_clusters > n_elements:
        raise ValueError('Number of clusters requested (%d) greater than '
                         'number of elements (%d)' % (n_clusters, n_elements))
    if clusterid is not None and len(clusterid) != n_elements:
        raise ValueError('clusterid must be None or an array of length '
                         'n_elements')
    if n_pass < 0:
        raise ValueError('n_pass must be greater than or equal to zero.')

    # setup distmatrix, which, for this library, is a double**
    # each distmatrix[i] needs to point to an array containing that row
    # of the lower-triangular distance matrix. So we need to convert
    # from upper-triangular format to lower triangular format.
    # (Note: the reason that this function even takes the upper triangular
    #  format as input is for better compatibilty with scipy)
    cdef double[::1] lower = triu_to_tril(condensed_distmatrix)
    cdef double ** distmatrix = <double**> malloc(n_elements * sizeof(double*))
    if not distmatrix:
      raise MemoryError()
    distmatrix[0] = NULL
    distmatrix[1] = &lower[0]
    for i in range(2, n_elements):
        distmatrix[i] = distmatrix[i-1] + (i - 1)

    # if DEBUG:
    #     from scipy.spatial.distance import squareform
    #     print('checking the distmatrix setup')
    #     square = squareform(condensed_distmatrix)
    #     for i in range(1, n_elements):
    #         for j in range(i):
    #             assert distmatrix[i][j] == square[i,j]
    #
    #     print('n_clusters = %d, n_elements=%d, n_pass=%d' % (
    #         n_clusters, n_elements, n_pass))

    # this is going to be the output, so we make a copy
    if clusterid is None:
        clusterid_ = np.zeros(n_elements, dtype=np.int32)
    else:
        clusterid_ = np.array(clusterid, dtype=np.int32, copy=True)

    # Call the underlying library
    _kmedoids(n_clusters, n_elements, distmatrix, n_pass,
              &clusterid_[0], &error, &ifound)
    free(distmatrix)

    return contigify_ids(clusterid_), error, ifound


# ========================================================================= #
# Utilities
# ========================================================================= #


def triu_to_tril(double[::1] upper):
    """Convert a distance matrix storage in upper-triangular format to lower
    triangular format.

    Upper triangular format stores the rows of the upper triangular portion
    of the (symmetric) matrix, (not including the diagonal) contiguously, and
    lower triangular format stores the rows of the lower triangular portion
    of matrix contiguously.

    This function converts from upper format to lower format.

    Example
    -------
    Upper triangular:

        [x   d(0,1)  d(0,2)  d(0,3)]
        [x     x     d(1,2)  d(1,3)]
        [x     x        x    d(2,3)]
        [x     x        x       x  ]

    The condensed upper triangular format (reading the rows across):

        [ d(0,1)  d(0,2)  d(0,3), d(1,2)  d(1,3), d(2,3) ]

    Lower triangular:
        [   x      x        x      x]
        [d(1,0)    x        x      x]
        [d(2,0)  d(2,1)     x      x]
        [d(3,0)  d(3,1)    d(3,2)  x]

    The condensed lower triangular format (reading the rows across):

        [d(1,0)  d(2,0)  d(2,1), d(3,0)  d(3,1), d(3,2)]
    """
    cdef double[::1] lower = np.zeros_like(upper)
    cdef int rvalue = _triu_to_tril(&upper[0], len(upper), &lower[0])
    if rvalue != 1:
        raise ValueError()
    return lower


def contigify_ids(int[::1] clusterids):
    """Renumber clusters to go from 0 to n_clusters - 1

    This function modifies the array inplace
    """
    _contigify_ids(&clusterids[0], len(clusterids))
    return np.array(clusterids, copy=False)
