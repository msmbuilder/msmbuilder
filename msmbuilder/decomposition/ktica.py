# Author:  Evan Feinberg<enf@stanford.edu>
# Contributors: Muneeb Sultan<msultan@stanford.edu>,
# Robert McGibbon <rmcgibbo@gmail.com>,
# Kyle A. Beauchamp  <kyleabeauchamp@gmail.com>
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
from .tica import tICA
from .kernal_approx import Nystroem
from ..base import BaseEstimator
from sklearn.pipeline import Pipeline
from ..utils import check_iter_of_sequences

__all__ = ['ktICA']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class ktICA(BaseEstimator):
    """ Convenience pipeline for doing kernal TICA. This method requires
    all the inputs for the Nystorm class and tica.

    Non-Linear dimensionality reduction using an eigendecomposition of the
    time-lag correlation matrix and covariance matrix of the Nystorm projection
    of the data.

    Parameters
    ----------
    Nystrorm Params
    k_name : string
        Name of the kernal. Default is 'rbf'
    k_n_components : int, default = 10
        Number of features to construct. How many data points will be used to
        construct the mapping.
    k_gamma: float, default=None
        Gamma parameter for the RBF, polynomial, exponential chi2 and sigmoid
        kernels. Interpretation of the default value is left to the kernel; see
        the documentation for sklearn.metrics.pairwise. Ignored by other
        kernels.

    Tica Params
    t_n_components : int, None
        Number of components to keep.
    t_lag_time : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+lag_time].
    t_gamma : nonnegative float, default=0.05
        Regularization strength. Positive `gamma` entails incrementing
        the sample covariance matrix by a constant times the identity,
        to ensure that it is positive definite. The exact form of the
        regularized sample covariance matrix is
        :math:`covariance + (gamma / n_features) * Tr(covariance) * Identity`
    t_weighted_transform : bool, default=False
        If True, weight the projections by the implied timescales, giving
        a quantity that has units [Time].

    Attributes
    ----------
    Kernel object
    -------------
    components  : array-like shape (k_n_components, n_features)
        Subset of training points used to construct the feature map.
    component_indices_ : array, shape (n_components)
        Indices of components_ in the training set.
    normalization_ : array, shape (n_components, n_components)
        Normalization matrix needed for embedding. Square root of the
        kernel matrix on k_components.
    TICA object
    -----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    offset_correlation_ : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix,
        :math:`C=E[(x_t)^T x_{t+lag}]`.
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
    analysis of molecular dynamics data in [1]_, [2]_, and [3]_. In [1]_ and
    [2]_, tICA was used as a dimensionality reduction technique before fitting
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

    def __init__(self, k_name='rbf', k_n_components=10, k_gamma=None,
                 t_n_components=None, t_lag_time=1, t_gamma=0.05,
                 t_weighted_transform=False):

        self._kernal__name = k_name
        self._kernal__components = k_n_components
        self._kernal__gamma = k_gamma
        self._tica__components = t_n_components
        self._tica__lagtime = t_lag_time
        self._tica__gamma = t_gamma
        self._tica__weighted_transform = t_weighted_transform

        self.nystroem = Nystroem(kernel=self._kernal__name,
                                 n_components=self._kernal__components,
                                 gamma=self._kernal__gamma)
        self.tica = tICA(n_components=self._tica__components,
                         lag_time=self._tica__lagtime,
                         gamma=self._tica__gamma,
                         weighted_transform=self._tica__weighted_transform)

        self._kt_pipeline = Pipeline(
            [('kernel', self.nystroem), ('tica', self.tica)])

    def fit(self, sequences, y=None):
        self._kt_pipeline.fit(sequences, y)
        return self

    def partial_fit(self, X):
        # I am not sure what parital fit even means within this context.
        raise NotImplementedError("Cant do ktica here")

    def transform(self, sequences):
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        sequences: list of array-like, each of shape (n_samples_i, n_features)
            Training data, where n_samples_i in the number of samples
            in sequence i and n_features is the number of features.

        Returns
        -------
        sequence_new : list of array-like, each of shape
        (n_samples_i, t_n_components) where t_n_components is the number of
        desired tic_components
        """
        check_iter_of_sequences(
            sequences, max_iter=3)  # we might be lazy-loading

        X_transformed = self._kt_pipeline.transform(sequences)

        return X_transformed

    def fit_transform(self, sequences, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        This method is not online. Any state accumulated from previous calls to
        `fit()`.

        Parameters
        ----------
        sequences: list of array-like, each of shape (n_samples_i, n_features)
            Training data, where n_samples_i in the number of samples
            in sequence i and n_features is the number of features.
        y : None
            Ignored

        Returns
        -------
        sequence_new : list of array-like, each of shape
        (n_samples_i, t_n_components) where t_n_components is the number of
        tic components to keep.
        """
        self.fit(sequences)
        return self.transform(sequences)
