# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors: Muneeb Sultan <msultan@stanford.edu>, Evan Feinberg <enf@stanford.edu>
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.

from __future__ import absolute_import

import numpy as np
from scipy.linalg import svd

from sklearn import kernel_approximation
from sklearn.metrics.pairwise import pairwise_kernels

from .base import MultiSequenceDecompositionMixin


__all__ = ['Nystroem', 'LandmarkNystroem']


class Nystroem(MultiSequenceDecompositionMixin, kernel_approximation.Nystroem):
    __doc__ = kernel_approximation.Nystroem.__doc__


class LandmarkNystroem(Nystroem):
    """Approximate a kernel map using a subset of the training data.

    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.
    Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

    Parameters
    ----------
    landmarks : ndarray of shape (n_frames, n_features)
        Custom landmark points for the Nyostroem approximation
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.
    n_components : int
        Number of features to construct.
        How many data points will be used to construct the mapping.
    gamma : float, default=None
        Gamma parameter for the RBF, polynomial, exponential chi2 and
        sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    random_state : {int, RandomState}, optional
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Subset of training points used to construct the feature map.
    component_indices_ : array, shape (n_components)
        Indices of ``components_`` in the training set.
    normalization_ : array, shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.

    References
    ----------
    .. [1] Williams, C.K.I. and Seeger, M.
       "Using the Nystroem method to speed up kernel machines",
       Advances in neural information processing systems 2001
    .. [2] T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
       "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
       Comparison",
       Advances in Neural Information Processing Systems 2012

    See also
    --------
    Nystroem : Approximate a kernel map using a subset of the training data.
    """

    def __init__(self, landmarks=None, **kwargs):
        if (landmarks is not None and
                not isinstance(landmarks, (int, np.ndarray))):
            raise ValueError('landmarks should be an int, ndarray, or None.')
        self.landmarks = landmarks
        super(LandmarkNystroem, self).__init__(**kwargs)

    def fit(self, sequences, y=None):
        if self.landmarks is not None:
            basis_kernel = pairwise_kernels(self.landmarks, metric=self.kernel,
                                            filter_params=True,
                                            **self._get_kernel_params())

            U, S, V = svd(basis_kernel)
            S = np.maximum(S, 1e-12)
            self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)
            self.components_ = self.landmarks
            self.component_indices_ = None

            return self

        super(Nystroem, self).fit(sequences, y=y)
