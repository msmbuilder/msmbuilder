# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
import numpy as np
import six
import scipy.spatial.distance
import warnings
from msmbuilder import libdistance
from scipy.cluster.hierarchy import fcluster
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin, TransformerMixin
from . import MultiSequenceClusterMixin
from ..base import BaseEstimator
from fastcluster import linkage

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------


__all__ = ['_LandmarkAgglomerative']


def ward_pooling_function(x, cluster_cardinality, intra_cluster_sum):
    normalization_factor = cluster_cardinality*(cluster_cardinality+1)/2
    squared_sums = (x**2).sum(axis=1)
    result_vector = ((cluster_cardinality * squared_sums -
                      intra_cluster_sum) / normalization_factor)
    return result_vector

POOLING_FUNCTIONS = {
    'average': lambda x, ignore1, ignore2: np.mean(x, axis=1),
    'complete': lambda x, ignore1, ignore2: np.max(x, axis=1),
    'single': lambda x, ignore1, ignore2: np.min(x, axis=1),
    'ward': ward_pooling_function,
}

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------


def pdist(X, metric='euclidean'):
    if isinstance(metric, six.string_types):
        return libdistance.pdist(X, metric)

    n = len(X)
    d = np.empty((n, n))
    for i in range(n):
        d[i, :] = metric(X, X, i)
    return scipy.spatial.distance.squareform(d, checks=False)


def cdist(XA, XB, metric='euclidean'):
    if isinstance(metric, six.string_types):
        return libdistance.cdist(XA, XB, metric)

    nA, nB = len(XA), len(XB)
    d = np.empty((nA, nB))
    for i in range(nA):
        d[i, :] = metric(XB, XA, i)
    return d


#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------


class _LandmarkAgglomerative(ClusterMixin, TransformerMixin):
    """Landmark-based agglomerative hierarchical clustering

    Landmark-based agglomerative clustering is a simple scalable version of
    "standard" hierarchical clustering which doesn't require computing the full
    matrix of pairwise distances between all data points. The idea is
    basically to subsample only ``n_landmarks`` "landmark"
    data points, cluster them, and then assign labels to the remaining data
    points based on their distances to (and the labels of) the landmarks.

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
    linkage : {'single', 'complete', 'average', 'ward'}, default='average'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
            - average uses the average of the distances of each observation of
              the two sets.
            - complete or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - single uses the minimum distance between all observations of the
              two sets.
            - ward linkage minimizes the within-cluster variance
        The linkage also effects the predict() method and the use of landmarks.
        After computing the distance from each new data point to the landmarks,
        the new data point will be assigned to the cluster that minimizes the
        linkage function between the new data point and each of the landmarks.
        (i.e with ``single``, new data points will be assigned the label of
        the closest landmark, with ``average``, it will be assigned the label
        of the landmark s.t. the mean distance from the test point to all the
        landmarks with that label is minimized, etc.)
    metric : string or callable, default= "euclidean"
        Metric used to compute the distance between samples.
    landmark_strategy : {'stride', 'random'}, default='stride'
        Method for determining landmark points. Only matters when n_landmarks
        is not None. "stride" takes landmarks every n-th data point in X, and
        random selects them  uniformly at random.
    random_state : integer or numpy.RandomState, optional
        The generator used to select random landmarks. Only used if
        landmark_strategy=='random'. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.
    max_landmarks : int, optional, default=None
        Useful for hyperparameter searching. If n_clusters exceeds n_landmarks,
        max_landmarks will be used. Otherwise, n_landmarks will be used. If
        None, no cutoff is enforced on n_landmarks, which may result in memory 
        issues.
    ward_predictor : {'single', 'complete', 'average', 'ward'}, default='ward'
        Which criterion to use when predicting cluster assignments after
        fitting with ward linkage.

    References
    ----------
    .. [1] Mullner, D. "Modern hierarchical, agglomerative clustering
        algorithms." arXiv:1109.2378 (2011).

    Attributes
    ----------
    landmark_labels_
    landmarks_
    """

    def __init__(self, n_clusters, n_landmarks=None, linkage='average',
                 metric='euclidean', landmark_strategy='stride',
                 random_state=None, max_landmarks=None, ward_predictor='ward'):
        self.n_clusters = n_clusters
        self.n_landmarks = n_landmarks
        self.metric = metric
        self.landmark_strategy = landmark_strategy
        self.random_state = random_state
        self.linkage = linkage
        self.max_landmarks = max_landmarks
        self.ward_predictor = ward_predictor

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

        if self.max_landmarks is not None:
            if self.n_clusters > self.n_landmarks:
                self.n_landmarks = self.max_landmarks
        
        if self.n_landmarks is None:
            distances = pdist(X, self.metric)
            tree = linkage(distances, method=self.linkage)
            self.landmark_labels_ = fcluster(tree, criterion='maxclust',
                                             t=self.n_clusters) - 1
            self.cardinality_ = np.bincount(self.landmark_labels_)
            self.squared_distances_within_cluster_ = np.zeros(self.n_clusters)

            n = len(X)
            for k in range(len(distances)):
                i = int(n - 2 - np.floor(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5))
                j = int(k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
                if self.landmark_labels_[i] == self.landmark_labels_[j]:
                    self.squared_distances_within_cluster_[
                        self.landmark_labels_[i]] += distances[k] ** 2

            self.landmarks_ = X

        else:
            if self.landmark_strategy == 'random':
                land_indices = check_random_state(self.random_state).randint(
                    len(X), size=self.n_landmarks)
            else:
                land_indices = np.arange(len(X))[::(len(X) //
                                        self.n_landmarks)][:self.n_landmarks]

            distances = pdist(X[land_indices], self.metric)
            tree = linkage(distances, method=self.linkage)
            self.landmark_labels_ = fcluster(tree, criterion='maxclust',
                                             t=self.n_clusters) - 1
            self.cardinality_ = np.bincount(self.landmark_labels_)
            self.squared_distances_within_cluster_ = np.zeros(self.n_clusters)

            n = len(X[land_indices])
            for k in range(len(distances)):
                i = int(n - 2 - np.floor(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5))
                j = int(k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
                if self.landmark_labels_[i] == self.landmark_labels_[j]:
                    self.squared_distances_within_cluster_[
                        self.landmark_labels_[i]] += distances[k] ** 2

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
        pfunc_name = self.ward_predictor if self.linkage == 'ward' else self.linkage

        try:
            pooling_func = POOLING_FUNCTIONS[pfunc_name]
        except KeyError:
                raise ValueError("linkage {} is not supported".format(pfunc_name))

        pooled_distances = np.empty(len(X))
        pooled_distances.fill(np.infty)
        labels = np.zeros(len(X), dtype=int)

        for i in range(self.n_clusters):
            if np.any(self.landmark_labels_ == i):
                d = pooling_func(dists[:, self.landmark_labels_ == i],
                                 self.cardinality_[i],
                                 self.squared_distances_within_cluster_[i])
                if np.any(d < 0):
                    warnings.warn("Distance shouldn't be negative.")
                mask = (d < pooled_distances)
                pooled_distances[mask] = d[mask]
                labels[mask] = i
            else:
                print("No data points were assigned to cluster {}".format(i))

        return labels

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        self.fit(X)
        return self.predict(X)


class LandmarkAgglomerative(MultiSequenceClusterMixin, _LandmarkAgglomerative,
                            BaseEstimator):
    __doc__ = _LandmarkAgglomerative.__doc__
    _allow_trajectory = True
