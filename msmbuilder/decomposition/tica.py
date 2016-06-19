# Author: Christian Schwantes <schwancr@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>, Kyle A. Beauchamp  <kyleabeauchamp@gmail.com>
# Copyright (c) 2014, Stanford University
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.linalg
import warnings
from ..base import BaseEstimator
from ..utils import check_iter_of_sequences, array2d
from sklearn.base import TransformerMixin

__all__ = ['tICA']

# -----------------------------------------------------------------------------
# Code
# -----------------------------------------------------------------------------


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
    shrinkage : float, default=None
        The covariance shrinkage intensity (range 0-1). If shrinkage is not
        specified (the default) it is estimated using an analytic formula
        (the Rao-Blackwellized Ledoit-Wolf estimator) introduced in [5].
    kinetic_mapping : bool, default=False
        If True, weigh the projections by the tICA eigenvalues, yielding
         kinetic distances as described in [6].

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
    n_observations_ : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences_ : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.
    timescales_ : array-like, shape (n_features,)
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
    .. [5] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. ICASSP (2009)
    .. [6] Noe, F. and Clementi, C. arXiv arXiv:1506.06259 [physics.comp-ph]
           (2015)
    """

    def __init__(self, n_components=None, lag_time=1, shrinkage=None,
                 kinetic_mapping=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.shrinkage = shrinkage
        self.shrinkage_ = None
        self.kinetic_mapping = kinetic_mapping

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
        self._eigenvalues_ = None

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
            # So, we're not dirty, but someone might have changed
            # n_components
            if len(self._eigenvalues_) >= self.n_components:
                return
            # if we've already solved for enough eigenvectors then
            # we don't need to solve it again

        # just check to make sure we've actually seen some data
        if self.n_observations_ == 0:
            raise RuntimeError('The model must be fit() before use.')

        lhs = self.offset_correlation_
        rhs = self.covariance_

        if not np.allclose(lhs, lhs.T):
            raise RuntimeError('offset correlation matrix is not symmetric')
        if not np.allclose(rhs, rhs.T):
            raise RuntimeError('correlation matrix is not symmetric')

        vals, vecs = scipy.linalg.eigh(lhs, b=rhs,
            eigvals=(self.n_features-self.n_components, self.n_features-1))

        # sort in order of decreasing value
        ind = np.argsort(vals)[::-1]
        vals = vals[ind]
        vecs = vecs[:, ind]

        self._eigenvalues_ = vals
        self._eigenvectors_ = vecs

        self._is_dirty = False

    @property
    def score_(self):
        """Training score of the model, computed as the generalized matrix,
        Rayleigh quotient, the sum of the first `n_components` eigenvalues
        """
        self._solve()
        return self._eigenvalues_[:self.n_components].sum()

    @property
    def eigenvectors_(self):
        self._solve()
        return self._eigenvectors_[:, :self.n_components]

    @property
    def eigenvalues_(self):
        self._solve()
        return self._eigenvalues_[:self.n_components]

    @property
    def timescales_(self):
        self._solve()
        return -1. * self.lag_time / np.log(self._eigenvalues_[:self.n_components])

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

        S = term - np.outer(means, means)  # sample covariance matix

        if self.shrinkage is None:
            sigma, self.shrinkage_ = rao_blackwell_ledoit_wolf(S, n=self.n_observations_)
        else:
            self.shrinkage_ = self.shrinkage
            p = self.n_features
            F = (np.trace(S) / p) * np.eye(p)  # shrinkage target
            sigma = (1-self.shrinkage)*S + self.shrinkage*F

        return sigma

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
        check_iter_of_sequences(sequences, max_iter=3)  # we might be lazy-loading
        for X in sequences:
            self._fit(X)

        if self.n_sequences_ == 0:
            raise ValueError('All sequences were shorter than '
                             'the lag time, %d' % self.lag_time)

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
        check_iter_of_sequences(sequences, max_iter=3)  # we might be lazy-loading
        sequences_new = []

        for X in sequences:
            X = array2d(X)
            if self.means_ is not None:
                X = X - self.means_
            X_transformed = np.dot(X, self.components_.T)

            if self.kinetic_mapping:
                X_transformed *= self.eigenvalues_

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

        # We don't need to scream and shout here. Just ignore this data.
        if not len(X) > self.lag_time:
            warnings.warn("length of data (%d) is too short for the lag time (%d)" % (len(X), self.lag_time))
            return

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
        sequences : list of array, each of shape (n_samples_i, n_features)
            Test data. A list of sequences in afeature space, each of which is a 2D
            array of possibily different lengths, but the same number of features.

        Returns
        -------
        gmrq : float
            Generalized matrix Rayleigh quotient. This number indicates how
            well the top ``n_timescales+1`` eigenvectors of this tICA model perform
            as slowly decorrelating collective variables for the new data in
            ``sequences``.

        References
        ----------
        .. [1] McGibbon, R. T. and V. S. Pande, "Variational cross-validation
           of slow dynamical modes in molecular kinetics" J. Chem. Phys. 142,
           124105 (2015)
        """

        assert self._initialized
        V = self.eigenvectors_

        # Note: How do we deal with regularization parameters like gamma
        # here? I'm not sure. Should C and S be estimated using self's
        # regularization parameters?
        m2 = self.__class__(shrinkage=self.shrinkage, n_components=self.n_components, lag_time=self.lag_time)
        for X in sequences:
            m2.partial_fit(X)

        numerator = V.T.dot(m2.offset_correlation_).dot(V)
        denominator = V.T.dot(m2.covariance_).dot(V)

        try:
            trace = np.trace(numerator.dot(np.linalg.inv(denominator)))
        except np.linalg.LinAlgError:
            trace = np.nan
        return trace


    def summarize(self):
        """Some summary information."""
        # force shrinkage to be calculated
        self.covariance_

        return """time-structure based Independent Components Analysis (tICA)
-----------------------------------------------------------
n_components        : {n_components}
shrinkage           : {shrinkage}
lag_time            : {lag_time}
kinetic_mapping     : {kinetic_mapping}

Top 5 timescales :
{timescales}

Top 5 eigenvalues :
{eigenvalues}
""".format(n_components=self.n_components, lag_time=self.lag_time,
           shrinkage=self.shrinkage_, kinetic_mapping=self.kinetic_mapping,
           timescales=self.timescales_[:5], eigenvalues=self.eigenvalues_[:5])


def rao_blackwell_ledoit_wolf(S, n):
    """Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance
    matrix.

    Parameters
    ----------
    S : array, shape=(n, n)
        Sample covariance matrix (e.g. estimated with np.cov(X.T))
    n : int
        Number of data points.

    Returns
    -------
    sigma : array, shape=(n, n)
    shrinkage : float

    References
    ----------
    .. [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
        estimation of high dimensional covariance matrices" ICASSP (2009)
    """
    p = len(S)
    assert S.shape == (p, p)

    alpha = (n-2)/(n*(n+2))
    beta = ((p+1)*n - 2) / (n*(n+2))

    trace_S2 = np.sum(S*S)  # np.trace(S.dot(S))
    U = ((p * trace_S2 / np.trace(S)**2) - 1)
    rho = min(alpha + beta/U, 1)

    F = (np.trace(S) / p) * np.eye(p)
    return (1-rho)*S + rho*F, rho
