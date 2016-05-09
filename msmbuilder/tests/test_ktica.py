from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_almost_equal

from ..decomposition import tICA, KernelTICA
from ..decomposition.kernel_approximation import LandmarkNystroem
from ..example_datasets import fetch_alanine_dipeptide
from ..featurizer import DihedralFeaturizer


def test_compare_to_tica():

    bunch = fetch_alanine_dipeptide()

    featurizer = DihedralFeaturizer(sincos=True)
    features = featurizer.transform(bunch['trajectories'][0:1])
    features = [features[0][::10]]

    tica = tICA(lag_time=1, n_components=2)
    ktica = KernelTICA(lag_time=1, kernel='linear', n_components=2,
                       random_state=42)

    tica_out = tica.fit_transform(features)[0]
    ktica_out = ktica.fit_transform(features)[0]

    assert_array_almost_equal(ktica_out, tica_out, decimal=1)


def test_compare_to_pipeline():
    np.random.seed(42)
    X = np.random.randn(100, 5)

    ktica = KernelTICA(kernel='rbf', lag_time=5, n_components=1,
                       random_state=42)
    y1 = ktica.fit_transform([X])[0]

    u = np.arange(X.shape[0])[5::1]
    v = np.arange(X.shape[0])[::1][:u.shape[0]]
    lndmrks = X[np.unique((u, v))]

    assert_array_almost_equal(lndmrks, ktica.landmarks, decimal=3)

    nystroem = LandmarkNystroem(kernel='rbf', landmarks=lndmrks,
                                random_state=42)
    tica = tICA(lag_time=5, n_components=1)

    y2_1 = nystroem.fit_transform([X])
    y2_2 = tica.fit_transform(y2_1)[0]

    assert_array_almost_equal(y1, y2_2)
