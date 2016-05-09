from __future__ import absolute_import

import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.metrics.pairwise import kernel_metrics

from ..decomposition import tICA
from ..decomposition.kernel_approx import Nystroem, LandmarkNystroem


def test_nystroem_approximation():
    pass
    # np.random.seed(42)
    # X = np.random.randn(10, 3)
    #
    # kernel = Nystroem(kernel='linear')
    # tica = tICA(n_components=2, lag_time=1)
    #
    # y2 = tica.fit_transform([np.copy(X)])[0]


def test_lndmrk_nystroem_approximation():
    pass
