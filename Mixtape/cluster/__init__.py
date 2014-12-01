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
from sklearn import cluster
from sklearn import mixture

import mdtraj as md
from ..base import BaseEstimator
from ..utils import check_iter_of_sequences

from .base import MultiSequenceClusterMixin
from .kcenters import KCenters
from .ndgrid import NDGrid
from .agglomerative import LandmarkAgglomerative
from .regularspatial import RegularSpatial
from .kmedoids import KMedoids
from .minibatchkmedoids import MiniBatchKMedoids

__all__ = ['KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
           'GMM', 'SpectralClustering', 'Ward', 'KCenters', 'NDGrid',
           'LandmarkAgglomerative', 'RegularSpatial', 'KMedoids',
           'MiniBatchKMedoids', 'MultiSequenceClusterMixin']

def _replace_labels(doc):
    """Really hacky find-and-replace method that modifies one of the sklearn
    docstrings to change the semantics of labels_ for the subclasses"""
    lines = doc.splitlines()
    labelstart, labelend = None, None
    foundattributes = False
    for i, line in enumerate(lines):
        if 'Attributes' in line:
            foundattributes = True
        if 'labels' in line and not labelstart and foundattributes:
            labelstart = len('\n'.join(lines[:i]))
        if labelstart and line.strip() == '' and not labelend:
            labelend = len('\n'.join(lines[:i+1]))


    replace  = '''\n    `labels_` : list of arrays, each of shape [sequence_length, ]
        The label of each point is an integer in [0, n_clusters).
    '''

    return doc[:labelstart] + replace + doc[labelend:]

#-----------------------------------------------------------------------------
# New "multisequence" versions of all of the clustering algorithims in sklearn
#-----------------------------------------------------------------------------

class KMeans(MultiSequenceClusterMixin, cluster.KMeans, BaseEstimator):
    __doc__ = _replace_labels(cluster.KMeans.__doc__)


class MiniBatchKMeans(MultiSequenceClusterMixin, cluster.MiniBatchKMeans, BaseEstimator):
    __doc__ = _replace_labels(cluster.MiniBatchKMeans.__doc__)

class AffinityPropagation(MultiSequenceClusterMixin, cluster.AffinityPropagation, BaseEstimator):
    __doc__ = _replace_labels(cluster.AffinityPropagation.__doc__)

class MeanShift(MultiSequenceClusterMixin, cluster.MeanShift, BaseEstimator):
    __doc__ = _replace_labels(cluster.MeanShift.__doc__)

class SpectralClustering(MultiSequenceClusterMixin, cluster.SpectralClustering, BaseEstimator):
    __doc__ = _replace_labels(cluster.SpectralClustering.__doc__)

class Ward(MultiSequenceClusterMixin, cluster.Ward, BaseEstimator):
    __doc__ = _replace_labels(cluster.Ward.__doc__)

class GMM(MultiSequenceClusterMixin, mixture.GMM, BaseEstimator):
    __doc__ = _replace_labels(mixture.GMM.__doc__)

