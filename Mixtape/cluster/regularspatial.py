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
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from mixtape.cluster import _regularspatialc
from mixtape.cluster import MultiSequenceClusterMixin

__all__ = ['RegularSpatial']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class _RegularSpatial(BaseEstimator, ClusterMixin, TransformerMixin):
    """Regular spatial clustering.

    Parameters
    ----------
    d_min : float
        Minimum distance between cluster centers. This parameter controls
        the number of clusters which are found.
    metric : string or function
        The distance metric to use. The distance function can
        be a callable (e.g. function). If it's callable, the
        function should have the signature shown below in
        the Notes. Alternatively, `metric` can be a string. In
        that case, it should be one of the metric strings
        accepted by scipy.spatial.distance.

    Notes
    -----
    Clusters are chosen to be approximately equally separated in conformation
    space with respect to the distance metric used. In pseudocode, the
    algorithm, from Senne et al., is:
      - Initialize a list of cluster centers containing only the first data
        point in the data set
      - Iterating over all conformations in the input dataset (in order),
          * If the data point is farther than d_min from all existing
            cluster center, add it to the list of cluster centers

    [Custom metrics] RegularSpatial can accept an arbitrary metric
    function. In the interest of performance, the expected call
    signature of a custom metric is

    >>> def mymetric(X, Y, yi):
        # return the distance from Y[yi] to each point in X.

    References
    ----------
    .. [1] Senne, Martin, et al. J. Chem Theory Comput. 8.7 (2012): 2223-2238

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    n_clusters_ : int
        The number of clusters located.
    """

    def __init__(self, d_min, metric='euclidean', opt=True):
        self.d_min = d_min
        self.metric = metric
        self.opt = opt

    def fit(self, X, y=None):
        if self.opt and self.metric == 'euclidean' and isinstance(X, np.ndarray):
            # fast path
            X = np.asarray(X, dtype=np.float64, order='c')
            self.cluster_centers_ = _regularspatialc._rspatial_euclidean(X, float(self.d_min))
            self.n_clusters_ = len(self.cluster_centers_)
            return self

        # regular code
        metric_function = self._metric_function
        if len(X) == 0:
            raise ValueError('len(X) must be greater than 0')

        self.cluster_centers_ = [X[0]]
        for i in range(1, len(X)):
            d = metric_function(np.array(self.cluster_centers_), X, i)
            if np.all(d > self.d_min):
                self.cluster_centers_.append(X[i])
        
        self.cluster_centers_ = np.array(self.cluster_centers_)
        self.n_clusters_ = len(self.cluster_centers_)
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

        for i in range(self.n_clusters_):
            d = metric_function(X, self.cluster_centers_, i)
            mask = (d < distances)
            distances[mask] = d[mask]
            labels[mask] = i

        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).predict(X)

    @property
    def _metric_function(self):
        if isinstance(self.metric, string_types):
            # distance from r[i] to each frame in t (output is a vector of length len(t)
            # using scipy.spatial.distance.cdist
            return lambda t, r, i : cdist(t, r[i, np.newaxis], metric=self.metric)[:,0]
        elif callable(self.metric):
            return self.metric
        raise NotImplementedError


class RegularSpatial(MultiSequenceClusterMixin, _RegularSpatial):
    __doc__ = _RegularSpatial.__doc__

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
        return self

