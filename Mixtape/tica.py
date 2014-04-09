# Author: Christian Schwantes
# Contributors: Robert McGibbon, Kyle Beauchamp
# Copyright (c) 2014, Stanford University
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

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.linalg
from sklearn.utils import array2d

__all__ = ['tICA']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class tICA(object):

    """Time-structure independent component analysis (tICA)

    Linear dimensionality reduction using an eigendecomposition of the
    time-lag correlation matrix and covariance matrix of the data and keeping
    only the vectors decorrelate slowest to project the data into a lower
    dimensional space.

    Parameters
    ----------
    n_components : int, None
        Number of components to keep.
    offset : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+offset].

    Attributes
    ----------
    `components_` : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    `offset_correlation_` : array-like, shape (n_features, n_features)
    `eigenvalues_` : array-like, shape (n_features,)
    `eigenvectors_` : array-like, shape (n_components, n_features)
    `means_` : array, shape (n_features,)
    `n_observations` : int
    `n_sequences` : int

    References
    ----------
    .. [1] Schwantes, Christian R., and Vijay S. Pande. J. Chem Theory Comput.
    9.4 (2013): 2000-2009.
    .. [2] Perez-Hernandez, Guillermo, et al. J Chem. Phys (2013): 015102.
    """

    def __init__(self, n_components=None, offset=1):
        self.n_components = n_components
        self.offset = offset

        self.n_features = None
        self.n_observations_ = None
        self.n_sequences_ = None

        self._initialized = False

        # X[:-self.offset].T dot X[self.offset:]
        self._outer_0_to_T_lagged = None
        # X[:-self.offset].sum(axis=0)
        self._sum_0_to_TminusOffset = None
        # X[self.offset:].sum(axis=0)
        self._sum_tau_to_T = None
        # X[self.offset:].sum(axis=0)
        self._sum_0_to_T = None

        # X[:-self.offset].T dot X[:-self.offset])
        self._outer_0_to_TminusOffset = None
        # X[self.offset:].T dot X[self.offset:]
        self._outer_offset_to_T = None

        # the tICs themselves
        self._components_ = None
        # Cached results of the eigendecompsition
        self._eigenvectors_ = None
        self._eigenvalues = None

        # are our current tICs ditry? this indicates that we've updated
        # the model with more data since the last time we computed components_,
        # eigenvalues, eigenvectors, and is set by _fit
        self._is_dirty = True

    def _initialize(self, n_features):
        if self._initialized:
            return

        if self.n_components is None:
            self.n_components = n_features
        self.n_features = n_features
        self.n_observations_ = 0
        self.n_sequences_ = 0
        self._outer_0_to_T_lagged = np.zeros((n_features, n_features))
        self._sum_0_to_TminusOffset = np.zeros(n_features)
        self._sum_tau_to_T = np.zeros(n_features)
        self._sum_0_to_T = np.zeros(n_features)
        self._outer_0_to_TminusOffset = np.zeros((n_features, n_features))
        self._outer_offset_to_T = np.zeros((n_features, n_features))

    def _solve_eigenproblem(self):
        if not self._is_dirty:
            return

        vals, vecs = scipy.linalg.eig(self.offset_correlation_, b=self.covariance_)

        # sort in order of decreasing value
        ind = np.argsort(np.real(vals))[::-1]
        vals = vals[ind]
        vecs = vecs[:, ind]

        self._eigenvalues_ = vals
        self._eigenvectors_ = vecs

        self._is_dirty = False

    @property
    def eigenvectors_(self):
        self._solve_eigenproblem()
        return self._eigenvectors_

    @property
    def eigenvalues_(self):
        self._solve_eigenproblem()
        return self._eigenvectors_

    @property
    def components_(self):
        return self.eigenvectors_[:, 0:self.n_components].T

    @property
    def means_(self):
        two_N = 2 * (self.n_observations_ - self.offset * self.n_sequences_)
        means = (self._sum_0_to_TminusOffset + self._sum_tau_to_T) / float(two_N)
        return means

    @property
    def offset_correlation_(self):
        two_N = 2 * (self.n_observations_ - self.offset * self.n_sequences_)
        term = (self._outer_0_to_T_lagged + self._outer_0_to_T_lagged.T) / two_N

        means = self.means_
        return term - np.outer(means, means)

    @property
    def covariance_(self):
        two_N = 2 * (self.n_observations_ - self.offset * self.n_sequences_)
        term = (self._outer_0_to_TminusOffset + self._outer_offset_to_T) / two_N

        means = self.means_
        return term - np.outer(means, means)

    def fit(self, X):
        """Fit the model with X.

        This method is not online.  Any state accumulated from previous calls to
        fit() or fit_update() will be cleared. For online learning, use
        `fit_ignore`.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._initialized = False
        self._fit(X)
        return self

    def partial_fit(self, X):
        """Fit the model with X.

        This method is suitable for online learning. The state of the model
        will be updated with the new data `X`.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        X = array2d(X)
        if self.means_ is not None:
            X = X - self.means_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.

        This method is not online. Any state accumulated from previous calls to
        fit() or fit_update() will be cleared. For online learning, use
        `fit_ignore`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

    def _fit(self, X):
        X = array2d(X)
        self._initialize(X.shape[1])

        self.n_observations_ += X.shape[0]
        self.n_sequences_ += 1

        self._outer_0_to_T_lagged += np.dot(X[:-self.offset].T, X[self.offset:])
        self._sum_0_to_TminusOffset += X[:-self.offset].sum(axis=0)
        self._sum_tau_to_T = X[self.offset:].sum(axis=0)
        self._sum_0_to_T = X.sum(axis=0)
        self._outer_0_to_TminusOffset = np.dot(X[:-self.offset].T, X[:-self.offset])
        self._outer_offset_to_T = np.dot(X[self.offset:].T, X[self.offset:])

        self._is_dirty = True
