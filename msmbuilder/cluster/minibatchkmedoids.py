# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division

from operator import itemgetter
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin, TransformerMixin

from . import MultiSequenceClusterMixin
from . import _kmedoids
from .. import libdistance
from ..base import BaseEstimator


class _MiniBatchKMedoids(ClusterMixin, TransformerMixin):
    """Mini-Batch K-Medoids clustering.

    This method finds a set of cluster centers that are themselves data points,
    attempting to minimize the mean-squared distance from the datapoints to
    their assigned cluster centers using only mini-batches of the dataset.

    Mini batches of the dataset are selected, and augmented to include each
    of the cluster centers. Then, standard KMedoids clustering is performed
    on the batch, using code based on the C clustering library [1]. The memory
    requirement scales as the square ``batch_size`` instead of the square of
    the size of the dataset.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, optional, default=5
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    batch_size : int, optional, default: 100
        Size of the mini batches.
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that sequences
        passed to ``fit()`` be ```md.Trajectory```; other distance metrics
        require ``np.ndarray``s.
    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that do not lead to any modified assignments.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    References
    ----------
    .. [1] de Hoon, Michiel JL, et al. "Open source clustering software."
       Bioinformatics 20.9 (2004): 1453-1454.

    See Also
    --------
    KMedoids:
        Batch version, requring O(N^2) memory.

    Attributes
    ----------
    cluster_ids_ : array, [n_clusters]
        Index of the data point that each cluster label corresponds to.
    cluster_centers_ : array, [n_clusters, n_features] or md.Trajectory
        Coordinates of cluster centers.
    labels_ : array, [n_samples,]
        The label of each point is an integer in [0, n_clusters).
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(self, n_clusters=8, max_iter=5, batch_size=100,
                 metric='euclidean', max_no_improvement=10, random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            if not (X.dtype == 'float32' or X.dtype == 'float64'):
                X = X.astype('float64')
        n_samples = len(X)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)
        random_state = check_random_state(self.random_state)

        cluster_ids_ = random_state.randint(0, n_samples, size=self.n_clusters)
        labels_ = random_state.randint(0, self.n_clusters, size=n_samples)

        n_iters_no_improvement = 0
        for kk in range(n_iter):
            # each minibatch includes the random indices AND the
            # current cluster centers
            minibatch_indices = np.concatenate([
                cluster_ids_,
                random_state.randint(0, n_samples, self.batch_size),
            ])
            dmat = libdistance.pdist(X, metric=self.metric, X_indices=np.array(minibatch_indices, dtype=np.intp))
            minibatch_labels = np.array(np.concatenate([
                np.arange(self.n_clusters),
                labels_[minibatch_indices[self.n_clusters:]]
            ]), dtype=np.intp)

            ids, intertia, _ = _kmedoids.kmedoids(
                self.n_clusters, dmat, 0, minibatch_labels,
                random_state=random_state)
            minibatch_labels, m = _kmedoids.contigify_ids(ids)

            # Copy back the new cluster_ids_ for the centers
            minibatch_cluster_ids = np.array(
                sorted(m.items(), key=itemgetter(1)))[:, 0]
            cluster_ids_ = minibatch_indices[minibatch_cluster_ids]

            # Copy back the new labels for the elements
            n_changed = np.sum(labels_[minibatch_indices] != minibatch_labels)
            if n_changed == 0:
                n_iters_no_improvement += 1
            else:
                labels_[minibatch_indices] = minibatch_labels
                n_iters_no_improvement = 0
            if n_iters_no_improvement >= self.max_no_improvement:
                break

        self.cluster_ids_ = cluster_ids_
        self.cluster_centers_ = X[cluster_ids_]
        self.labels_, self.inertia_ = libdistance.assign_nearest(
            X, self.cluster_centers_, metric=self.metric)
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
        if isinstance(X, np.ndarray):
            if not (X.dtype == 'float32' or X.dtype == 'float64'):
                X = X.astype('float64')
        labels, inertia = libdistance.assign_nearest(
            X, self.cluster_centers_, metric=self.metric)
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_


class MiniBatchKMedoids(MultiSequenceClusterMixin, _MiniBatchKMedoids, BaseEstimator):
    _allow_trajectory = True
    __doc__ = _MiniBatchKMedoids.__doc__[: _MiniBatchKMedoids.__doc__.find('Attributes')] + \
    '''
    Attributes
    ----------
    `cluster_centers_` : array, [n_clusters, n_features]
        Coordinates of cluster centers

    `labels_` : list of arrays, each of shape [sequence_length, ]
        `labels_[i]` is an array of the labels of each point in
        sequence `i`. The label of each point is an integer in
        [0, n_clusters).
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
        self.cluster_ids_ = self._split_indices(self.cluster_ids_)
        return self

    def summarize(self):
        return """MiniBatchKMedoids clustering
----------------------------
n_clusters : {n_clusters}
metric     : {metric}

Inertia    : {inertia_}
""".format(**self.__dict__)
