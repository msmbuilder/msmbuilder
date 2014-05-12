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
import numbers
import numpy as np
from sklearn.utils import array2d
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from mixtape.cluster import MultiSequenceClusterMixin

__all__ = ['NDGrid']
EPS = 1e-10

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class _NDGrid(BaseEstimator, ClusterMixin, TransformerMixin):
    """Discretize continuous data points onto an N-dimensional
    grid.

    This is in some sense the zero-th order approximation to
    clustering. We throw down an n-dimensional cartesian grid
    over the data points and then quantize each data point by
    the index of the bin it's in.

    Parameters
    ----------
    n_bins_per_feature : int
        Number of bins along each feature (degree of freedom) the total
        number of bins will be :math:`n_bins^{n_features}`.
    min : {float, array-like, None}, optional
        Lower bin edge. If None (default), the min and max for each feature
        will be fit during training.
    max : {float, array-like, None}, optional
        Upper bin edge. If None (default), the min and max for each feature
        will be fit during training.

    Attributes
    ----------
    n_features : int
        Number of features
    n_bins : int
        The total number of bins
    grid : np.ndarray, shape=[n_features, n_bins_per_feature+1]
        Bin edges
    """

    def __init__(self, n_bins_per_feature=2, min=None, max=None):
        self.n_bins_per_feature = n_bins_per_feature
        self.min = min
        self.max = max
        # unknown until we have the number of features
        self.n_features = None
        self.n_bins = None
        self.grid = None

    def fit(self, X, y=None):
        """Fit the grid

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data points

        Returns
        -------
        self
        """
        X = array2d(X)
        self.n_features = X.shape[1]
        self.n_bins = self.n_bins_per_feature**self.n_features

        if self.min is None:
            min = np.min(X, axis=0)
        elif isinstance(self.min, numbers.Number):
            min = self.min * np.ones(self.n_features)
        else:
            min = np.asarray(self.min)
            if not min.shape == (self.n_features,):
                raise ValueError('min shape error')

        if self.max is None:
            max = np.max(X, axis=0)
        elif isinstance(self.max, numbers.Number):
            max = self.max * np.ones(self.n_features)
        else:
            max = np.asarray(self.max)
            if not max.shape == (self.n_features,):
                raise ValueError('max shape error')
        
        self.grid = np.array([np.linspace(min[i]-EPS, max[i]+EPS, self.n_bins_per_feature + 1) for i in range(self.n_features)])

        return self

    def predict(self, X):
        """Get the index of the grid cell containing each sample in X
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data
        
        Returns
        -------
        y : array, shape = [n_samples,]
            Index of the grid cell containing each sample
        """
        if np.any(X < self.grid[:, 0]) or np.any(X > self.grid[:, -1]):
            raise ValueError('data out of min/max bounds')

        binassign = np.zeros((self.n_features, len(X)), dtype=int)
        for i in range(self.n_features):
            binassign[i] = np.digitize(X[:, i], self.grid[i]) - 1
        labels = np.dot(self.n_features**np.arange(self.n_features), binassign)

        assert np.max(labels) < self.n_bins
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    
class NDGrid(MultiSequenceClusterMixin, _NDGrid):
    __doc__ = _NDGrid.__doc__
