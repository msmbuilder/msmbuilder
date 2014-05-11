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
import six
import scipy.spatial.distance
from scipy.cluster.hierarchy import fcluster
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from mixtape.cluster import MultiSequenceClusterMixin

try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------


__all__ = ['_LandmarkAgglomerative']

POOLING_FUNCTIONS = {
    'average': lambda x: np.mean(x, axis=1),
    'complete': lambda x: np.max(x, axis=1),
    'single': lambda x: np.min(x, axis=1),
}


#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------


def pdist(X, metric='euclidean'):
    if isinstance(metric, six.string_types):
        return scipy.spatial.distance.pdist(X, metric)

    n = len(X)
    d = np.empty((n, n))
    for i in range(n):
        d[i, :] = metric(X, X, i)
    return scipy.spatial.distance.squareform(d, checks=False)


def cdist(XA, XB, metric='euclidean'):
    if isinstance(metric, six.string_types):
        return scipy.spatial.distance.cdist(XA, XB, metric)

    nA, nB = len(XA), len(XB)
    d = np.empty((nA, nB))
    for i in range(nA):
        d[i, :] = metric(XB, XA, i)
    return d


#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------


class _LandmarkAgglomerative(BaseEstimator, ClusterMixin, TransformerMixin):
    """Landmark-based agglomerative hierarchical clustering
    
    Parameters
    ----------
    n_clusters : int
        The number of clusters to find.
    n_landmarks : int, optional
        Memory-saving approximation. Instead of actually clustering every
        point, we instead select n_landmark points either randomly or by
        striding the data matrix (see ``landmark_strategy``). Then we cluster
        the only the landmarks, and then assign the remaining dataset based
        on distances to the landmarks. Note that n_landmarks=None is equivalent
        to using every point in the dataset as a landmark.
    linkage : {'single', 'complete', 'average'}, default='average'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
            - average uses the average of the distances of each observation of
              the two sets.
            - complete or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - single uses the minimum distance between all observations of the
              two sets.
        The linkage also effects the predict() method and the use of landmarks.
        After computing the distance from each new data point to the landmarks,
        the new data point will be assigned to the cluster that minimizes the
        linkage function between the new data point and each of the landmarks.
        (i.e with ``single``, new data points will be assigned the label of
        the closest landmark, with ``average``, it will be assigned the label
        of the landmark s.t. the mean distance from the test point to all the
        landmarks with that label is minimized, etc.)
    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the distance matrix.
    metric : string or callable, default= "euclidean"
        Metric used to compute the distance between samples.
    landmark_strategy : {'stride', 'random'}, default='stride'
        Method for determining landmark points. Only matters when n_landmarks
        is not None. "stride" takes landmarks every n-th data point in X, and
        random selects them randomly.
    random_state : integer or numpy.RandomState, optional
        The generator used to select random landmarks. Only used if
        landmark_strategy=='random'. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.

    Attributes
    ----------
    landmark_labels_
    landmarks_
    """
    
    def __init__(self, n_clusters, n_landmarks=None, linkage='average',
                 memory=Memory(cachedir=None, verbose=0), metric='euclidean',
                 landmark_strategy='stride', random_state=None):
        self.n_clusters = n_clusters
        self.n_landmarks = n_landmarks
        self.memory = memory
        self.metric = metric
        self.landmark_strategy = landmark_strategy
        self.random_state = random_state
        self.linkage = linkage

        self.landmark_labels_ = None
        self.landmarks_ = None

    def fit(self, X, y=None):
        """
        Compute agglomerative clustering.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
        
        Returns
        -------
        self
        """

        memory = self.memory
        if isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)
        if self.n_landmarks is None:
            distances = memory.cache(pdist)(X, self.metric)
        else:
            if self.landmark_strategy == 'random':
                land_indices = check_random_state(self.random_state).randint(len(X), size=self.n_landmarks)
            else:
                land_indices = np.arange(len(X))[::(len(X)/self.n_landmarks)][:self.n_landmarks]
            distances = memory.cache(pdist)(X[land_indices], self.metric)

        tree = memory.cache(linkage)(distances, method=self.linkage)
        self.landmark_labels_ = fcluster(tree, criterion='maxclust', t=self.n_clusters) - 1
        
        if self.n_landmarks is None:
            self.landmarks_ = X
        else:
            self.landmarks_ = X[land_indices]

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        dists = cdist(X, self.landmarks_, self.metric)

        try:
            pooling_func = POOLING_FUNCTIONS[self.linkage]
        except KeyError:
            raise ValueError('linkage=%s is not supported' % self.linkage)

        pooled_distances = np.empty(len(X))
        pooled_distances.fill(np.infty)
        labels = np.zeros(len(X), dtype=int)

        for i in range(self.n_clusters):
            if np.any(self.landmark_labels_ == i):
                d = pooling_func(dists[:, self.landmark_labels_ == i])
                mask = (d < pooled_distances)
                pooled_distances[mask] = d[mask]
                labels[mask] = i

        return labels

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        self.fit(X)
        return self.predict(X)


class LandmarkAgglomerative(MultiSequenceClusterMixin, _LandmarkAgglomerative):
    __doc__ = _LandmarkAgglomerative.__doc__
