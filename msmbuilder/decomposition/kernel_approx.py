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
from ..utils import check_iter_of_sequences


__all__ = ['Nystroem', 'LandmarkNystroem']


class Nystroem(MultiSequenceDecompositionMixin, kernel_approximation.Nystroem):
    __doc__ = kernel_approximation.Nystroem.__doc__


class LandmarkNystroem(Nystroem):
    __doc__ = Nystroem.__doc__

    def __init__(self, basis=None, **kwargs):
        if basis is not None and not isinstance(basis, (int, np.ndarray)):
            raise ValueError('basis should be an int, ndarray, or None.')
        self.basis = basis
        super(LandmarkNystroem, self).__init__(**kwargs)

    def fit(self, sequences, y=None):
        check_iter_of_sequences(sequences)
        X = self._concat(sequences)
        if self.basis is not None:
            self.components_ = self.basis

            basis_kernel = pairwise_kernels(self.basis, metric=self.kernel,
                                            filter_params=True,
                                            **self._get_kernel_params())

            U, S, V = svd(basis_kernel)
            S = np.maximum(S, 1e-12)
            self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)

            return self

        super(Nystroem, self).fit(X, y=y)
