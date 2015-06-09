# Author: Matthew Harrigan <matthew.p.harrigan@gmail.com>
# Contributors:
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.

from __future__ import absolute_import, print_function, division
from scipy.spatial import KDTree as sp_KDTree
import numpy as np

from . import check_iter_of_sequences


class KDTree(object):
    # Code lovingly copied from MultiSequenceClusterMixin

    _allow_trajectory = False

    def __init__(self, sequences):
        check_iter_of_sequences(sequences,
                                allow_trajectory=self._allow_trajectory)
        self._kdtree = sp_KDTree(self._concat(sequences))

    def query(self, x, k=1, p=2, distance_upper_bound=np.inf):
        cdists, cinds = self._kdtree.query(x, k, p, distance_upper_bound)
        return cdists, self._split_indices(cinds)

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
