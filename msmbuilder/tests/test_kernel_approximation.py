from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.kernel_approximation import Nystroem as NystroemR

from msmbuilder.decomposition.kernel_approximation import Nystroem, LandmarkNystroem


def test_nystroem_vs_sklearn():
    np.random.seed(42)
    X = np.random.randn(100, 5)

    kernel = Nystroem(kernel='linear', random_state=42)
    kernelR = NystroemR(kernel='linear', random_state=42)

    y1 = kernel.fit_transform([X])[0]
    y2 = kernelR.fit_transform(X)

    assert_array_almost_equal(y1, y2)


def test_lndmrk_nystroem_approximation():
    np.random.seed(42)
    X = np.random.randn(100, 5)

    u = np.arange(X.shape[0])[5::1]
    v = np.arange(X.shape[0])[::1][:u.shape[0]]
    lndmrks = X[np.unique((u, v))]

    kernel = LandmarkNystroem(kernel='rbf', random_state=42)
    kernelR = NystroemR(kernel='rbf', random_state=42)

    y1_1 = kernel.fit_transform([X])[0]
    kernel.landmarks = lndmrks
    y1_2 = kernel.fit_transform([X])[0]

    y2 = kernelR.fit_transform(X)

    assert_array_almost_equal(y2, y1_1)

    assert not all((np.abs(y2 - y1_2) > 1E-6).flatten())
