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

from operator import itemgetter
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin, TransformerMixin
from . import _kmedoids


class _KMedoids(ClusterMixin, TransformerMixin):
    """K-Medoids clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to be found.
    n_passes : int
        The number of times clustering is performed. Clustering is performed
        n_passes times, each time starting from a different (random) initial
        assignment.
    metric : string, default='euclidean'
        The distance metric to use, it should be one of the metric strings
        accepted by scipy.spatial.distance.

    Attributes
    ----------
    cluster_ids_ : array, [n_clusters]
        Index of the data point that each cluster label corresponds to.
    labels_ : array, [n_samples,]
        The label of each point is an integer in [0, n_clusters).
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(self, n_clusters=8, n_passes=5, metric='euclidean'):
        self.n_clusters = n_clusters
        self.n_passes = n_passes
        self.metric = metric

    def fit(self, X, y=None):
        if self.n_passes < 1:
            raise ValueError('n_passes must be greater than 0. got %s' %
                             self.n_passes)
        if self.n_clusters < 1:
            raise ValueError('n_passes must be greater than 0. got %s' %
                             self.n_clusters)

        dmat = pdist(X, metric=self.metric)
        ids, self.intertia_, _ = _kmedoids.kmedoids(
            self.n_clusters, dmat, self.n_passes)

        self.labels_, mapping = _kmedoids.contigify_ids(ids)
        smapping = sorted(mapping.items(), key=itemgetter(1))
        self.cluster_ids_ = np.array(smapping)[:, 0]

        return self


class _MinibatchKMedoids(ClusterMixin, TransformerMixin):
    """

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, optional
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    batch_size : int, optional, default: 100
        Size of the mini batches.

    Attributes
    ----------
    cluster_ids_ : array, [n_clusters]
        Index of the data point that each cluster label corresponds to.
    labels_ : array, [n_samples,]
        The label of each point is an integer in [0, n_clusters).
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(self, n_clusters=8, max_iter=100, batch_size=100,
                 metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        n_samples = len(X)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)
        random_state = check_random_state(self.random_state)

        cid = random_state.random_integers(
            low=0, high=self.n_clusters-1, size=n_samples)
        unique_labels = random_state.random_integers(
            low=0, high=n_samples-1, size=self.n_clusters)
        labels = unique_labels[cid].astype(np.int32)

        for kk in range(n_iter):
            # each minibatch includes the random indices AND the
            # current cluster centers
            _, mapping = _kmedoids.contigify_ids(np.copy(labels))
            vmapping = np.array(sorted(mapping.items(), key=itemgetter(1)))

            minibatch_indices = np.concatenate([
                vmapping[:, 0],
                random_state.random_integers(
                    0, n_samples - 1, self.batch_size),
            ])

            dmat = pdist(X[minibatch_indices], metric=self.metric)

            minibatch_labels = np.vectorize(mapping.get, otypes=[np.int32])(
                labels[minibatch_indices])

            # print('\nminibatch indices', minibatch_indices)
            # print('mapping', mapping)
            # print('labels', labels[minibatch_indices])
            # print('mapped labels', minibatch_labels)
            assert np.all(minibatch_labels[np.arange(self.n_clusters)] ==
                          np.arange(self.n_clusters))

            ids, intertia, _ = _kmedoids.kmedoids(
                self.n_clusters, dmat, 0, minibatch_labels)
            # print('output ids', ids)

            # mapping from original centers to new centers, from the begining
            # of the output
            new_centers = dict((vmapping[i, 0], minibatch_indices[ids[i]])
                               for i in range(self.n_clusters))
            # perform the mapping to the whole labels array
            labels = np.vectorize(new_centers.get, otypes=[np.int32])(labels)
            # manually map the elements that were in the minibatch to their
            # new labels
            labels[minibatch_indices[self.n_clusters:]] = \
                minibatch_indices[ids[self.n_clusters:]]

            unique_labels = np.unique(labels)
            assert np.all(labels[unique_labels] == unique_labels)

            from scipy.spatial.distance import euclidean
            print('intertia', sum(euclidean(X[i], X[labels[i]]) for i in range(n_samples)))

        self.labels_, mapping = _kmedoids.contigify_ids(labels)
        smapping = sorted(mapping.items(), key=itemgetter(1))
        self.cluster_ids_ = np.array(smapping)[:, 0]
        self.intertia_ = 0  # TODO()

        return self
