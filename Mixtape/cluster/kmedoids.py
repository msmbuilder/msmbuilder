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

import operator
import numpy as np
from scipy.spatial.distance import pdist
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
        n_pass times, each time starting from a different (random) initial
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
        smapping = sorted(mapping.items(), key=operator.itemgetter(1))
        self.cluster_ids_ = np.array(smapping)[:, 0]

        return self