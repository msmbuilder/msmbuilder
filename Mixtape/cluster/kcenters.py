# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2013, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
import numpy as np
from six import string_types, PY2
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster.k_means import _squared_norms, _labels_inertia

from mixtape.cluster import MultiSequenceClusterMixin

__all__ = ['KCenters']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class _KCenters(BaseEstimator, ClusterMixin):
    """

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
    [Custom metrics] KCenters can accept an arbitrary metric
    function. In the interest of performance, the expected call
    signature of a custom metric is

    >>> def mymetric(target_sequence, ref_sequence, ref_index):
        # return the distance from ref_sequence[ref_index] to each
        # data point in target_sequence.

    [Algorithim] KCenters is a simple clustering algorithm. To
    initialize, we select a random data point to be the first
    cluster center. In each iteration, we maintain knowledge of
    the distance from each data point to its assigned cluster center
    (the nearest cluster ceneter). In the iteration, we increase the
    number of cluster centers by one by choosing the data point which
    is farthest from its assigned cluster center to be the new
    cluster cluster.

    Attributes
    ----------
    `cluster_centers_` : array, [n_clusters, n_features]
        Coordinates of cluster centers

    `labels_` : array, [n_samples,]
        Labels of each point. The label of each point is
        an integer in [0, n_clusters).

    `distances_` : array, [n_samples,]
        Distance from each sample to the cluster center it is
        assigned to.
    """
    def __init__(self, n_clusters=8, metric='euclidean', random_state=0):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random = check_random_state(random_state)

        if isinstance(metric, string_types):
            self.metric_function = lambda target_sequence, ref_sequence, ref_index : \
                                   cdist(target_sequence, ref_sequence[ref_index, np.newaxis], metric=metric)[:,0]
        elif callable(metric):
            self.metric_function = metric

    def fit(self, X, y=None):
        n_samples = len(X)
        new_center_index = self.random.randint(0, n_samples)

        self.labels_ = np.zeros(n_samples, dtype=int)
        self.distances_ = np.empty(n_samples, dtype=float)
        self.distances_.fill(np.inf)

        if isinstance(self.metric, string_types):
            self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
        else:
            # this should be a list, not a numpy array, so that
            # fit() works if X is a non-numpyarray collection type
            # like an molecular dynamics trajectory with a non-string metric
            self.cluster_centers_ = [None for i in range(self.n_clusters)]

        for i in range(self.n_clusters):
            d = self.metric_function(X, X, new_center_index)
            mask = (d < self.distances_)
            self.distances_[mask] = d[mask]
            self.labels_[mask] = i
            self.cluster_centers_[i] = X[new_center_index]

            new_center_index = np.argmax(self.distances_)

        return self

    def predict(self, X):
        if self.metric == 'euclidean':
            x_squared_norms = _squared_norms(X)
            return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]

        # todo: assignments
        raise NotImplementedError('havent gotten around to this yet')

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_


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
        lengths = [len(s) for s in sequences]
        self.distances_ = self._split(self.distances_, lengths)
        return self
