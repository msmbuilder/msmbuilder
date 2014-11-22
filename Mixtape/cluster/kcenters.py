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

from __future__ import absolute_import, print_function, division
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin, TransformerMixin

from .. import libdistance
from . import MultiSequenceClusterMixin
from ..base import BaseEstimator

__all__ = ['KCenters']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class _KCenters(ClusterMixin, TransformerMixin):
    """K-Centers clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that sequences
        passed to ``fit()`` be ```md.Trajectory```; other distance metrics
        require ``np.ndarray``s.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Attributes
    ----------
    cluster_ids_ : array, [n_clusters]
        Index of the data point that each cluster label corresponds to.
    cluster_centers_ : array, [n_clusters, n_features] or md.Trajectory
        Coordinates of cluster centers
    labels_ : array, [n_samples,]
        The label of each point is an integer in [0, n_clusters).
    distances_ : array, [n_samples,]
        Distance from each sample to the cluster center it is
        assigned to.
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(self, n_clusters=8, metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        n_samples = len(X)
        new_center_index = check_random_state(self.random_state).randint(0, n_samples)

        self.labels_ = np.zeros(n_samples, dtype=int)
        self.distances_ = np.empty(n_samples, dtype=float)
        self.distances_.fill(np.inf)
        cluster_ids_ = []

        for i in range(self.n_clusters):
            d = libdistance.dist(X, X[new_center_index], metric=self.metric)
            mask = (d < self.distances_)
            self.distances_[mask] = d[mask]
            self.labels_[mask] = i
            cluster_ids_.append(new_center_index)
            new_center_index = np.argmax(self.distances_)

        self.cluster_ids_ = cluster_ids_
        self.cluster_centers_ = X[cluster_ids_]
        self.inertia_ = np.sum(self.distances_)
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
        return self.fit(X, y).labels_


class KCenters(MultiSequenceClusterMixin, _KCenters, BaseEstimator):
    _allow_trajectory = True
    __doc__ = _KCenters.__doc__[: _KCenters.__doc__.find('Attributes')] + \
    '''
    Attributes
    ----------
    `cluster_centers_` : array, [n_clusters, n_features]
        Coordinates of cluster centers

    `labels_` : list of arrays, each of shape [sequence_length, ]
        `labels_[i]` is an array of the labels of each point in
        sequence `i`. The label of each point is an integer in
        [0, n_clusters).

    `distances_` : list of arrays, each of shape [sequence_length, ]
        `distances_[i]` is an array of  the labels of each point in
        sequence `i`. Distance from each sample to the cluster center
        it is assigned to.
    '''

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
        self.distances_ = self._split(self.distances_)
        return self

    def summarize(self):
        return """
KCenters clustering
--------------------
n_clusters : {n_clusters}
metric     : {metric}

Inertia       : {inertia}
Mean distance : {mean_distance}
Max  distance : {max_distance}
""".format(n_clusters=self.n_clusters, metric=self.metric,
           inertia=self.inertia_, mean_distance=np.mean(np.concatenate(self.distances_)),
           max_distance=np.max(np.concatenate(self.distances_)))
