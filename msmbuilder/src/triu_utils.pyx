"""
Utilities related to indexing upper triangular matrices with a diagonal
offset of 1. The semantics match ``numpy.triu_indices(n, k=1)``
"""
from numpy cimport npy_intp


cdef inline npy_intp ij_to_k(npy_intp i, npy_intp j, npy_intp n) nogil:
    """2D (i, j) square matrix index to linearized upper triangular index

    [ 0  a0  a1  a2  a3 ]    (i=0,j=1) -> 0
    [ 0   0  a4  a5  a6 ]    (i=0,j=2) -> 1
    [ 0   0   0  a7  a8 ]    (i=1,j=3) -> 5
    [ 0   0   0   0  a9 ]       etc
    [ 0   0   0   0   0 ]    (i=4,j=5) -> 9

    For further explanation, see http://stackoverflow.com/a/27088560/1079728

    Parameters
    ----------
    i : int
        Row index
    j : int
        Column index
    n : int
        Matrix size. The matrix is assumed to be square

    Returns
    -------
    k : int
        Linearized upper triangular index

    See Also
    --------
    k_to_ij : the inverse operation
    """
    if j > i:
        return (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
    return (n*(n-1)/2) - (n-j)*((n-j)-1)/2 + i - j - 1


cdef inline void k_to_ij(npy_intp k, npy_intp n, npy_intp *i, npy_intp *j) nogil:
    """Linearized upper triangular index to 2D (i, j) index

    [ 0  a0  a1  a2  a3 ]      0 -> (i=0,j=1)
    [ 0   0  a4  a5  a6 ]      1 -> (i=0,j=2)
    [ 0   0   0  a7  a8 ]      5 -> (i=1,j=3)
    [ 0   0   0   0  a9 ]            etc
    [ 0   0   0   0   0 ]

      http://stackoverflow.com/a/27088560/1079728

    Parameters
    ----------
    k : int
        Linearized upper triangular index

    Returns
    -------
    i : int
        Row index, written into *i on exit
    j : int
        Column index, written into *j on exit
    """

    i[0] = n - 2 - <int>(sqrt(-8.0*k + 4.0*n*(n-1)-7.0)/2.0 - 0.5)
    j[0] = k + i[0] + 1 - n*(n-1)/2 + (n-i[0])*(n-i[0]-1)/2