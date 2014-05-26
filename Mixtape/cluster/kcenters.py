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
from six import string_types, PY2
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from mixtape.cluster import MultiSequenceClusterMixin

__all__ = ['KCenters']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class _KCenters(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Centers clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    metric : string or function
        The distance metric to use. The distance function can
        be a callable (e.g. function). If it's callable, the
        function should have the signature shown below in
        the Notes. Alternatively, `metric` can be a string. In
        that case, it should be one of the metric strings
        accepted by scipy.spatial.distance.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Notes
    -----
    K-Centers one of the most inexpensive possible clustering algorithms. It's
    sometimes also called "max-min" clustering. The algorithm stats with a
    single data point as a cluster center, and then at each iteration it chooses
    as the next cluster center the data point which is farthest from its
    assigned cluster center.

    [Custom metrics] KCenters can accept an arbitrary metric
    function. In the interest of performance, the expected call
    signature of a custom metric is

    >>> def mymetric(X, Y, yi):
        # return the distance from Y[yi] to each point in X.

    [Algorithm] KCenters is a simple clustering algorithm. To
    initialize, we select a random data point to be the first
    cluster center. In each iteration, we maintain knowledge of
    the distance from each data point to its assigned cluster center
    (the nearest cluster center). In the iteration, we increase the
    number of cluster centers by one by choosing the data point which
    is farthest from its assigned cluster center to be the new
    cluster cluster.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ : array, [n_samples,]
        The label of each point is an integer in [0, n_clusters).
    distances_ : array, [n_samples,]
        Distance from each sample to the cluster center it is
        assigned to.
    """
    def __init__(self, n_clusters=8, metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.random = check_random_state(random_state)

    def fit(self, X, y=None):
        n_samples = len(X)
        new_center_index = self.random.randint(0, n_samples)

        self.labels_ = np.zeros(n_samples, dtype=int)
        self.distances_ = np.empty(n_samples, dtype=float)
        self.distances_.fill(np.inf)

        metric_function = self._metric_function

        if isinstance(self.metric, string_types):
            self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
        else:
            # this should be a list, not a numpy array, so that
            # fit() works if X is a non-numpyarray collection type
            # like an molecular dynamics trajectory with a non-string metric
            self.cluster_centers_ = [None for i in range(self.n_clusters)]

        for i in range(self.n_clusters):
            d = metric_function(X, X, new_center_index)
            mask = (d < self.distances_)
            self.distances_[mask] = d[mask]
            self.labels_[mask] = i
            self.cluster_centers_[i] = X[new_center_index]
            new_center_index = np.argmax(self.distances_)

        if not isinstance(self.metric, string_types):
            if isinstance(self.cluster_centers_[0], np.ndarray):
                self.cluster_centers_ = np.concatenate(self.cluster_centers_)
            else:
                # this is a hack to make md.trajectory work using
                # metric=md.rmsd
                self.cluster_centers_ = self.cluster_centers_[0].join(self.cluster_centers_[1:])

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
        metric_function = self._metric_function

        labels = np.zeros(len(X), dtype=int)
        distances = np.empty(len(X), dtype=float)
        distances.fill(np.inf)

        for i in range(self.n_clusters):
            d = metric_function(X, self.cluster_centers_, i)
            mask = (d < distances)
            distances[mask] = d[mask]
            labels[mask] = i

        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    @property
    def _metric_function(self):
        if isinstance(self.metric, string_types):
            # distance from r[i] to each frame in t (output is a vector of length len(t)
            # using scipy.spatial.distance.cdist
            return lambda t, r, i : cdist(t, r[i, np.newaxis], metric=self.metric)[:,0]
        elif callable(self.metric):
            return self.metric
        raise NotImplementedError


class KCenters(MultiSequenceClusterMixin, _KCenters):
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
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        self
        """
        MultiSequenceClusterMixin.fit(self, sequences)
        self.distances_ = self._split(self.distances_)
        return self

