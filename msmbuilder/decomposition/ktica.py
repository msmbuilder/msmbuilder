# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors: Christian Schwantes <schwancr@gmail.com>
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np

from .tica import tICA
from .kernel_approx import LandmarkNystroem


class KernelTICA(tICA):
    """Time-structure Independent Componenent Analysis (tICA) using the kernel trick.

    The kernel trick allows one to extend a linear method (e.g. tICA) to
    include non-linear solutions.

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
    weighted_transform : bool, default=False
        Deprecated. Please use kinetic_mapping.
    kinetic_mapping : bool, default=False
        If True, weigh the projections by the tICA eigenvalues, yielding
         kinetic distances as described in [6].
    kernel : str or callable
        Kernel map to be approximated using the Nystroem approximation.
        It must be one of:
            - 'linear' : linear kernel (dot product in the input space)
            - 'poly' : polynomial kernel
            - 'rbf' : radial basis function
            - 'sigmoid' : sigmoid kernel
            - ‘laplacian’ : laplacian kernel
            - 'cosine' : cosine similarity kernel
            - callable : A callable should accept two arguments
            and the keyword arguments passed to this object as kernel_params,
            and should return a floating point number.
    degree : int, optional
        Degree of the polynomial kernel. This is only used if kernel
        is 'poly'.
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If gamma == 0.0,
        then will use 1 / n_features.
    coef0 : float, optional
        Independent term in 'poly' and 'sigmoid'.
    basis : ndarray, optional
        Custom basis for Nyostroem approximation
    stride : int, optional
        Only sample pairs of points from the data according to this stride
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

    References
    ----------
    .. [1] Schwantes, Christian R. and Vijay S. Pande. "Modeling Molecular
           Kinetics with tICA and the Kernel Trick." In Review. (2014)
    """
    def __init__(self, kernel='rbf', degree=3, gamma=None, coef0=1., stride=1,
                 basis=None, **kwargs):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.stride = 1
        self.basis = basis
        self.kernel_params = {
                              'kernel': self.kernel,
                              'degree': self.degree,
                              'gamma': self.gamma,
                              'coef0': self.coef0,
                              }
        super(KernelTICA, self).__init__(**kwargs)

    def _gen_basis(self, sequences):
        X_0 = []
        X_t = []
        for seq in sequences:
            seq_t = seq[self.lag_time::self.stride]
            seq_0 = seq[::self.stride][:seq_t.shape[0]]
            X_0.append(seq_0)
            X_t.append(seq_t)
        return np.concatenate([np.concatenate(X_0), np.concatenate(X_t)])

    def fit(self, sequences, y=None):
        if self.basis is None:
            self.basis = self._gen_basis(sequences)
        self._nystroem = LandmarkNystroem(basis=self.basis,
                                          **self.kernel_params)
        self._nystroem.fit(sequences)
        super(KernelTICA, self).fit(sequences, y=y)

    def partial_fit(self, X):
        if self.basis is None:
            self.basis = self._gen_basis([X])
        if self._nystroem is None:
            self._nystroem = LandmarkNystroem(basis=self.basis,
                                              **self.kernel_params)
            self._nystroem.partial_fit(X)

        self._fit(self.kernel.partial_transform(X))
        return self
