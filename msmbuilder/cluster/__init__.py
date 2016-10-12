# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Matthew Harrigan <matthew.harrigan@outlook.com>, Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.


from __future__ import absolute_import, print_function, division
import warnings

from sklearn import cluster

try:
    # sklearn >= 0.18
    from sklearn.mixture import GaussianMixture as sklearn_GMM
except ImportError:
    from sklearn.mixture import GMM as sklearn_GMM

from ..base import BaseEstimator
from .base import MultiSequenceClusterMixin
from .kcenters import KCenters
from .ndgrid import NDGrid
from .agglomerative import LandmarkAgglomerative
from .regularspatial import RegularSpatial
from .kmedoids import KMedoids
from .minibatchkmedoids import MiniBatchKMedoids
from .apm import APM

warnings.filterwarnings("once", '', DeprecationWarning, r'^sklearn\.')

__all__ = ['KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
           'GMM', 'SpectralClustering', 'KCenters', 'NDGrid',
           'LandmarkAgglomerative', 'RegularSpatial', 'KMedoids',
           'MiniBatchKMedoids', 'MultiSequenceClusterMixin', 'APM',
           'AgglomerativeClustering']


def _replace_labels(doc):
    """Really hacky find-and-replace method that modifies one of the sklearn
    docstrings to change the semantics of labels_ for the subclasses"""
    lines = doc.splitlines()
    labelstart, labelend = None, None
    foundattributes = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Attributes':
            foundattributes = True
        if foundattributes and not labelstart and stripped.startswith('labels_'):
            labelstart = len('\n'.join(lines[:i])) + 1
        if labelstart and not labelend and stripped == '':
            labelend = len('\n'.join(lines[:i + 1]))

    if labelstart is None or labelend is None:
        return doc

    replace = '\n'.join([
        '    labels_ : list of arrays, each of shape [sequence_length, ]',
        '        The label of each point is an integer in [0, n_clusters).',
        '',
    ])
    return doc[:labelstart] + replace + doc[labelend:]


class KMeans(MultiSequenceClusterMixin, cluster.KMeans, BaseEstimator):
    __doc__ = _replace_labels(cluster.KMeans.__doc__)


class MiniBatchKMeans(MultiSequenceClusterMixin, cluster.MiniBatchKMeans,
                      BaseEstimator):
    __doc__ = _replace_labels(cluster.MiniBatchKMeans.__doc__)


class AffinityPropagation(MultiSequenceClusterMixin,
                          cluster.AffinityPropagation, BaseEstimator):
    __doc__ = _replace_labels(cluster.AffinityPropagation.__doc__)


class MeanShift(MultiSequenceClusterMixin, cluster.MeanShift, BaseEstimator):
    __doc__ = _replace_labels(cluster.MeanShift.__doc__)


class SpectralClustering(MultiSequenceClusterMixin, cluster.SpectralClustering,
                         BaseEstimator):
    __doc__ = _replace_labels(cluster.SpectralClustering.__doc__)


class AgglomerativeClustering(MultiSequenceClusterMixin,
                              cluster.AgglomerativeClustering,
                              BaseEstimator):
    __doc__ = _replace_labels(cluster.AgglomerativeClustering.__doc__)


class GMM(MultiSequenceClusterMixin, sklearn_GMM, BaseEstimator):
    __doc__ = _replace_labels(sklearn_GMM.__doc__)
