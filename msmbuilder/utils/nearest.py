# Author: Matthew Harrigan <matthew.p.harrigan@gmail.com>
# Contributors:
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.

from __future__ import absolute_import, print_function, division
from scipy.spatial import KDTree as sp_KDTree
import numpy as np

from . import check_iter_of_sequences


class KDTree(object):
    """kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points which
    can be used to rapidly look up the nearest neighbors of any point.
    This class wraps sklearn's implementation by taking a list of arrays
    and returning indices of the form (traj_i, frame_i).

    Parameters
    ----------
    sequences : list of (N,K) array_like
        Each array contains data points to be indexed. This array is not
        copied, and so modifying this data will result in bogus results.
    leafsize : int, optional
        The number of points at which the algorithm switches over to
        brute-force. Has to be positive.

    Raises
    ------
    RuntimeError
        The maximum recursion limit can be exceeded for large data
        sets.  If this happens, either increase the value for the `leafsize`
        parameter or increase the recursion limit by::
            >>> import sys
            >>> sys.setrecursionlimit(10000)

    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    The tree also supports all-neighbors queries, both with arrays of points
    and with other kd-trees. These do use a reasonably efficient algorithm,
    but the kd-tree is not necessarily the best data structure for this
    sort of calculation.
    """

    _allow_trajectory = False

    def __init__(self, sequences, leafsize=10):
        check_iter_of_sequences(sequences,
                                allow_trajectory=self._allow_trajectory)
        self._kdtree = sp_KDTree(self._concat(sequences), leafsize=leafsize)

    def query(self, x, k=1, p=2, distance_upper_bound=np.inf):
        """Query the kd-tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : int, optional
            The number of nearest neighbors to return.
        eps : nonnegative float, optional
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float, optional
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple if
            k is one, or tuple+(k,) if k is larger than one. Missing
            neighbors (e.g. when k > n or distance_upper_bound is
            given) are indicated with infinite distances.  If k is None,
            then d is an object array of shape tuple, containing lists
            of distances. In either case the hits are sorted by distance
            (nearest first).
        i : tuple(int, int) or array of tuple(int, int)
            The locations of the neighbors in self.data. Locations are
            given by tuples of (traj_i, frame_i)

        Examples
        --------
        >>> from msmbuilder.utils import KDTree
        >>> X1 = 0.3 * np.random.RandomState(0).randn(500, 2)
        >>> X2 = 0.3 * np.random.RandomState(1).randn(1000, 2) + 10
        >>> tree = KDTree([X1, X2])
        >>> pts = np.array([[0, 0], [10, 10]])
        >>> tree.query(pts)
        (array([ 0.0034,  0.0102]), array([[  0, 410], [  1, 670]]))
        >>> tree.query(pts[0])
        (0.0034, array([  0, 410]))
        """
        cdists, cinds = self._kdtree.query(x, k, p, distance_upper_bound)
        return cdists, self._split_indices(cinds)

    # concat and split code lovingly copied from MultiSequenceClusterMixin
    def _concat(self, sequences):
        self.__lengths = [len(s) for s in sequences]
        if len(sequences) > 0 and isinstance(sequences[0], np.ndarray):
            concat = np.ascontiguousarray(np.concatenate(sequences))
        else:
            raise TypeError('sequences must be a list of numpy arrays')

        assert sum(self.__lengths) == len(concat)
        return concat

    def _split(self, concat):
        return [concat[cl - l: cl] for (cl, l) in
                zip(np.cumsum(self.__lengths), self.__lengths)]

    def _split_indices(self, concat_inds):
        """Take indices in 'concatenated space' and return as pairs
        of (traj_i, frame_i)
        """
        clengths = np.append([0], np.cumsum(self.__lengths))
        mapping = np.zeros((clengths[-1], 2), dtype=int)
        for traj_i, (start, end) in enumerate(zip(clengths[:-1], clengths[1:])):
            mapping[start:end, 0] = traj_i
            mapping[start:end, 1] = np.arange(end - start)
        return mapping[concat_inds]
