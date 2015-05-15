# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin

from .. import libdistance
from . import MultiSequenceClusterMixin
from ..base import BaseEstimator

__all__ = ['RegularSpatial']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class _RegularSpatial(ClusterMixin, TransformerMixin):
    """Regular spatial clustering.

    This method finds a set of cluster centers that are themselves data points,
    chosen to be approximately equally separated in conformation space with
    respect to the distance metric used. In pseudocode, the
    algorithm, from Senne et al. [1], is:

      - Initialize a list of cluster centers containing only the first data
        point in the data set
      - Iterating over all conformations in the input dataset (in order),
          * If the data point is farther than ``d_min`` from all existing
            cluster center, add it to the list of cluster centers

    Parameters
    ----------
    d_min : float
        Minimum distance between cluster centers. This parameter controls
        the number of clusters which are found.
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that sequences
        passed to ``fit()`` be ```md.Trajectory```; other distance metrics
        require ``np.ndarray``s.

    References
    ----------
    .. [1] Senne, Martin, et al. J. Chem Theory Comput. 8.7 (2012): 2223-2238

    Attributes
    ----------
    cluster_center_indices_: array, [n_clusters]
        Indices of the positions chosen as cluster centers. Each entry is a
        (trajectory_index, frame_index) pair.
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    n_clusters_ : int
        The number of clusters located.
    """

    def __init__(self, d_min, metric='euclidean'):
        self.d_min = d_min
        self.metric = metric

    def fit(self, X, y=None):
        cluster_ids = [0]
        for i in range(1, len(X)):
            # distance from X[i] to each X with indices in cluster_ids
            d = libdistance.dist(
                X, X[i], metric=self.metric, X_indices=np.array(cluster_ids, dtype=np.intp))
            if np.all(d > self.d_min):
                cluster_ids.append(i)

        self.cluster_center_indices_ = cluster_ids
        self.cluster_centers_ = X[np.array(cluster_ids)]
        self.n_clusters_ = len(cluster_ids)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        Y : array, shape [n_samples,]
            Index of the closest center each sample belongs to.
        """
        labels, inertia = libdistance.assign_nearest(
            X, self.cluster_centers_, metric=self.metric)
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).predict(X)


class RegularSpatial(MultiSequenceClusterMixin, _RegularSpatial, BaseEstimator):
    __doc__ = _RegularSpatial.__doc__
    _allow_trajectory = True

    def fit(self, sequences, y=None):
        """Fit the kcenters clustering on the data

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries, or ``md.Trajectory``. Each
            sequence may have a different length, but they all must have the
            same number of features, or the same number of atoms if they are
            ``md.Trajectory``s.

        Returns
        -------
        self
        """
        MultiSequenceClusterMixin.fit(self, sequences)
        self.cluster_center_indices_ = self._split_indices(self.cluster_center_indices_)
        return self

    def summarize(self):
        return """
RegularSpatial clustering
-------------------------
d_min      : {d_min}
metric     : {metric}

n_clusters : {n_clusters}
""".format(d_min=self.d_min, metric=self.metric,
           n_clusters=self.n_clusters_)
