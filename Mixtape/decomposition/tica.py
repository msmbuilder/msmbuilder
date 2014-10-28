# Author: Christian Schwantes <schwancr@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>, Kyle A. Beauchamp  <kyleabeauchamp@gmail.com>
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

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import array2d

__all__ = ['tICA']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class tICA(BaseEstimator, TransformerMixin):

    """Time-structure Independent Component Analysis (tICA)

    Linear dimensionality reduction using an eigendecomposition of the
    time-lag correlation matrix and covariance matrix of the data and keeping
    only the vectors which decorrelate slowest to project the data into a lower
    dimensional space.

    Parameters
    ----------
    n_components : int, None
        Number of components to keep.
    lag_time : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+lag_time].
    gamma : nonnegative float, default=0.05
        Regularization strength. Positive `gamma` entails incrementing
        the sample covariance matrix by a constant times the identity,
        to ensure that it is positive definite. The exact form of the
        regularized sample covariance matrix is
        :math:`covariance + (gamma / n_features) * Tr(covariance) * Identity`
    weighted_transform : bool, default=False
        If True, weight the projections by the implied timescales, giving
        a quantity that has units [Time].

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    offset_correlation_ : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix, :math:`C=E[(x_t)^T x_{t+lag}]`.
    eigenvalues_ : array-like, shape (n_features,)
        Eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.
    eigenvectors_ : array-like, shape (n_components, n_features)
        Eigenvectors of the tICA generalized eigenproblem. The vectors
        give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium. Each eigenvector
        is associated with characteritic timescale
        :math:`- \frac{lag_time}{ln \lambda_i}, where :math:`lambda_i` is
        the corresponding eigenvector. See [2] for more information.
    means_ : array, shape (n_features,)
        The mean of the data along each feature
    n_observations : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.
    timescales : array-like, shape (n_features,)
        The implied timescales of the tICA model, given by 
        -offset / log(eigenvalues)

    Notes
    -----
    This method was introduced originally in [4]_, and has been applied to the
    analysis of molecular dynamics data in [1]_, [2]_, and [3]_. In [1]_ and [2]_,
    tICA was used as a dimensionality reduction technique before fitting
    other kinetic models.


    References
    ----------
    .. [1] Schwantes, Christian R., and Vijay S. Pande. J.
       Chem Theory Comput. 9.4 (2013): 2000-2009.
    .. [2] Perez-Hernandez, Guillermo, et al. J Chem. Phys (2013): 015102.
    .. [3] Naritomi, Yusuke, and Sotaro Fuchigami. J. Chem. Phys. 134.6
       (2011): 065101.
    .. [4] Molgedey, Lutz, and Heinz Georg Schuster. Phys. Rev. Lett. 72.23
       (1994): 3634.
    """
    
    def __init__(self, n_components=None, lag_time=1, gamma=0.05, weighted_transform=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.gamma = gamma
        self.weighted_transform = weighted_transform

        self.n_features = None
        self.n_observations_ = None
        self.n_sequences_ = None

        self._initialized = False

        # X[:-self.lag_time].T dot X[self.lag_time:]
        self._outer_0_to_T_lagged = None
        # X[:-self.lag_time].sum(axis=0)
        self._sum_0_to_TminusTau = None
        # X[self.lag_time:].sum(axis=0)
        self._sum_tau_to_T = None
        # X[:].sum(axis=0)
        self._sum_0_to_T = None

        # X[:-self.lag_time].T dot X[:-self.lag_time])
        self._outer_0_to_TminusTau = None
        # X[self.lag_time:].T dot X[self.lag_time:]
        self._outer_offset_to_T = None

        # the tICs themselves
        self._components_ = None
        # Cached results of the eigendecompsition
        self._eigenvectors_ = None
        self._eigenvalues = None

        # are our current tICs dirty? this indicates that we've updated
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
        self._sum_0_to_TminusTau = np.zeros(n_features)
        self._sum_tau_to_T = np.zeros(n_features)
        self._sum_0_to_T = np.zeros(n_features)
        self._outer_0_to_TminusTau = np.zeros((n_features, n_features))
        self._outer_offset_to_T = np.zeros((n_features, n_features))
        self._initialized = True

    def _solve(self):
        if not self._is_dirty:
            return

        if not np.allclose(self.offset_correlation_, self.offset_correlation_.T):
            raise RuntimeError('offset correlation matrix is not symmetric')
        if not np.allclose(self.covariance_, self.covariance_.T):
            raise RuntimeError('correlation matrix is not symmetric')

        rhs = self.covariance_ + (self.gamma / self.n_features) * \
                np.trace(self.covariance_) * np.eye(self.n_features)
        vals, vecs = scipy.linalg.eigh(self.offset_correlation_, b=rhs,
            eigvals=(self.n_features-self.n_components, self.n_features-1))

        # sort in order of decreasing value
        ind = np.argsort(vals)[::-1]
        vals = vals[ind]
        vecs = vecs[:, ind]

        self._eigenvalues_ = vals
        self._eigenvectors_ = vecs

        self._is_dirty = False

    @property
    def eigenvectors_(self):
        self._solve()
        return self._eigenvectors_

    @property
    def eigenvalues_(self):
        self._solve()
        return self._eigenvalues_

    @property
    def timescales_(self):
        self._solve()
        return -1. * self.lag_time / np.log(self._eigenvalues_)

    @property
    def components_(self):
        return self.eigenvectors_[:, 0:self.n_components].T

    @property
    def means_(self):
        two_N = 2 * (self.n_observations_ - self.lag_time * self.n_sequences_)
        means = (self._sum_0_to_TminusTau + self._sum_tau_to_T) / float(two_N)
        return means

    @property
    def offset_correlation_(self):
        two_N = 2 * (self.n_observations_ - self.lag_time * self.n_sequences_)
        term = (self._outer_0_to_T_lagged + self._outer_0_to_T_lagged.T) / two_N

        means = self.means_
        value = term - np.outer(means, means)
        return value

    @property
    def covariance_(self):
        two_N = 2 * (self.n_observations_ - self.lag_time * self.n_sequences_)
        term = (self._outer_0_to_TminusTau + self._outer_offset_to_T) / two_N

        means = self.means_
        value = term - np.outer(means, means)
        return value

    def fit(self, sequences, y=None):
        """Fit the model with a collection of sequences.

        This method is not online.  Any state accumulated from previous calls to
        fit() or partial_fit() will be cleared. For online learning, use
        `partial_fit`.

        Parameters
        ----------
        sequences: list of array-like, each of shape (n_samples_i, n_features)
            Training data, where n_samples_i in the number of samples
            in sequence i and n_features is the number of features.
        y : None
            Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._initialized = False
        for X in sequences:
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

    def transform(self, sequences):
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        sequences: list of array-like, each of shape (n_samples_i, n_features)
            Training data, where n_samples_i in the number of samples
            in sequence i and n_features is the number of features.

        Returns
        -------
        sequence_new : list of array-like, each of shape (n_samples_i, n_components)

        """
        sequences_new = []

        for X in sequences:
            X = array2d(X)
            if self.means_ is not None:
                X = X - self.means_
            X_transformed = np.dot(X, self.components_.T)

            if self.weighted_transform:        
                X_transformed *= self.timescales_

            sequences_new.append(X_transformed)

        return sequences_new

    def partial_transform(self, features):
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        features: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.  This function
            acts on a single featurized trajectory.

        Returns
        -------
        sequence_new : array-like, shape (n_samples, n_components)
            TICA-projected features
            
        Notes
        -----
        This function acts on a single featurized trajectory. 

        """
        sequences = [features]
        return self.transform(sequences)[0]

    def fit_transform(self, sequences, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        This method is not online. Any state accumulated from previous calls to
        `fit()` or `partial_fit()` will be cleared. For online learning, use
        `partial_fit`.

        Parameters
        ----------
        sequences: list of array-like, each of shape (n_samples_i, n_features)
            Training data, where n_samples_i in the number of samples
            in sequence i and n_features is the number of features.
        y : None
            Ignored

        Returns
        -------
        sequence_new : list of array-like, each of shape (n_samples_i, n_components)
        """
        self.fit(sequences)
        return self.transform(sequences)

    def _fit(self, X):
        X = np.asarray(array2d(X), dtype=np.float64)
        self._initialize(X.shape[1])
        if not len(X) > self.lag_time:
            raise ValueError('First dimension must be longer than '
                'lag_time=%d. X has shape (%d, %d)' % ((self.lag_time,) + X.shape))

        self.n_observations_ += X.shape[0]
        self.n_sequences_ += 1

        self._outer_0_to_T_lagged += np.dot(X[:-self.lag_time].T, X[self.lag_time:])
        self._sum_0_to_TminusTau += X[:-self.lag_time].sum(axis=0)
        self._sum_tau_to_T += X[self.lag_time:].sum(axis=0)
        self._sum_0_to_T += X.sum(axis=0)
        self._outer_0_to_TminusTau += np.dot(X[:-self.lag_time].T, X[:-self.lag_time])
        self._outer_offset_to_T += np.dot(X[self.lag_time:].T, X[self.lag_time:])

        self._is_dirty = True

    def score(self, sequences, y=None):
        """Score the model on new data using the generalized matrix Rayleigh quotient

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        Returns
        -------
        gmrq : float
            Generalized matrix Rayleigh quotient. This number indicates how
            well the top ``n_timescales+1`` eigenvectors of this MSM perform as
            slowly decorrelating collective variables for the new data in
            ``sequences``.

        References
        ----------
        .. [1] McGibbon, R. T. and V. S. Pande, "Variational cross-validation
           of slow dynamical modes in molecular kinetics"
           http://arxiv.org/abs/1407.8083 (2014)
        """

        assert self._initialized
        V = self.eigenvectors_

        # Note: How do we deal with regularization parameters like gamma
        # here? I'm not sure. Should C and S be estimated using self's
        # regularization parameters?
        m2 = self.__class__(lag_time=self.lag_time)
        for X in sequences:
            m2.partial_fit(X)

        numerator = V.T.dot(m2.offset_correlation_).dot(V)
        denominator = V.T.dot(m2.covariance_).dot(V)

        try:
            trace = np.trace(numerator.dot(np.linalg.inv(denominator)))
        except np.linalg.LinAlgError:
            trace = np.nan
        return trace
