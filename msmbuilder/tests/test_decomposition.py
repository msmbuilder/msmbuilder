from __future__ import absolute_import

import numpy as np
from mdtraj.testing import eq
from numpy.testing import assert_approx_equal
from numpy.testing import assert_array_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA as PCAr

from msmbuilder.example_datasets import AlanineDipeptide
from ..cluster import KCenters
from ..decomposition import (FactorAnalysis, FastICA, KernelTICA,
                             MiniBatchSparsePCA, PCA, SparsePCA, tICA)
from ..decomposition.kernel_approximation import LandmarkNystroem
from ..featurizer import DihedralFeaturizer

random = np.random.RandomState(42)
trajs = [random.randn(10, 3) for _ in range(5)]


def test_tica_fit_transform():
    X = random.randn(10, 3)

    tica = tICA(n_components=2, lag_time=1)
    y2 = tica.fit_transform([np.copy(X)])[0]


def test_tica_singular_1():
    tica = tICA(n_components=1)

    # make some data that has one column repeated twice
    X = random.randn(100, 2)
    X = np.hstack((X, X[:, 0, np.newaxis]))

    tica.fit([X])
    assert tica.components_.dtype == np.float64
    assert tica.eigenvalues_.dtype == np.float64


def test_tica_singular_2():
    tica = tICA(n_components=1)

    # make some data that has one column of all zeros
    X = random.randn(100, 2)
    X = np.hstack((X, np.zeros((100, 1))))

    tica.fit([X])
    assert tica.components_.dtype == np.float64
    assert tica.eigenvalues_.dtype == np.float64


def test_tica_shape():
    model = tICA(n_components=3).fit([random.randn(100, 10)])
    eq(model.eigenvalues_.shape, (3,))
    eq(model.eigenvectors_.shape, (10, 3))
    eq(model.components_.shape, (3, 10))


def test_tica_score_1():
    X = random.randn(100, 5)
    for n in range(1, 5):
        tica = tICA(n_components=n, shrinkage=0)
        tica.fit([X])
        assert_approx_equal(
            tica.score([X]),
            tica.eigenvalues_.sum())
        assert_approx_equal(tica.score([X]), tica.score_)


def test_tica_score_2():
    X = random.randn(100, 5)
    Y = random.randn(100, 5)
    model = tICA(shrinkage=0.0, n_components=2).fit([X])
    s1 = model.score([Y])
    s2 = tICA(shrinkage=0.0).fit(model.transform([Y])).eigenvalues_.sum()

    eq(s1, s2)


def test_tica_multiple_components():
    X = random.randn(100, 5)
    tica = tICA(n_components=1, shrinkage=0)
    tica.fit([X])

    Y1 = tica.transform([X])[0]

    tica.n_components = 4
    Y4 = tica.transform([X])[0]

    tica.n_components = 3
    Y3 = tica.transform([X])[0]

    assert Y1.shape == (100, 1)
    assert Y4.shape == (100, 4)
    assert Y3.shape == (100, 3)

    eq(Y1.flatten(), Y3[:, 0])
    eq(Y3, Y4[:, :3])


def test_tica_kinetic_mapping():
    X = random.randn(10, 3)

    tica1 = tICA(n_components=2, lag_time=1)
    tica2 = tICA(n_components=2, lag_time=1, kinetic_mapping=True)

    y1 = tica1.fit_transform([np.copy(X)])[0]
    y2 = tica2.fit_transform([np.copy(X)])[0]

    assert eq(y2, y1 * tica1.eigenvalues_)


def test_pca_vs_sklearn():
    # Compare msmbuilder.pca with sklearn.decomposition

    pcar = PCAr()
    pcar.fit(np.concatenate(trajs))

    pca = PCA()
    pca.fit(trajs)

    y_ref1 = pcar.transform(trajs[0])
    y1 = pca.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)
    np.testing.assert_array_almost_equal(pca.components_, pcar.components_)
    np.testing.assert_array_almost_equal(pca.explained_variance_,
                                         pcar.explained_variance_)
    np.testing.assert_array_almost_equal(pca.mean_, pcar.mean_)
    np.testing.assert_array_almost_equal(pca.n_components_, pcar.n_components_)
    np.testing.assert_array_almost_equal(pca.noise_variance_,
                                         pcar.noise_variance_)


def test_pca_pipeline():
    # Test that PCA it works in a msmbuilder pipeline

    p = Pipeline([('pca', PCA()), ('cluster', KCenters())])
    p.fit(trajs)


def test_pca_generator():
    # Check to see if it works with a generator

    traj_dict = dict((i, t) for i, t in enumerate(trajs))

    pcar = PCAr()
    pcar.fit(np.concatenate(trajs))

    pca = PCA()
    # on python 3, dict.values() returns a generator
    pca.fit(traj_dict.values())

    y_ref1 = pcar.transform(trajs[0])
    y1 = pca.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)
    np.testing.assert_array_almost_equal(pca.components_, pcar.components_)
    np.testing.assert_array_almost_equal(pca.explained_variance_,
                                         pcar.explained_variance_)
    np.testing.assert_array_almost_equal(pca.mean_, pcar.mean_)
    np.testing.assert_array_almost_equal(pca.n_components_, pcar.n_components_)
    np.testing.assert_array_almost_equal(pca.noise_variance_,
                                         pcar.noise_variance_)


def test_sparsepca():
    pca = SparsePCA()
    pca.fit_transform(trajs)
    pca.summarize()


def test_minibatchsparsepca():
    pca = MiniBatchSparsePCA()
    pca.fit_transform(trajs)
    pca.summarize()


def test_fastica():
    ica = FastICA()
    ica.fit_transform(trajs)
    ica.summarize()


def test_factoranalysis():
    fa = FactorAnalysis()
    fa.fit_transform(trajs)
    fa.summarize()


def test_ktica_compare_to_tica():
    trajectories = AlanineDipeptide().get_cached().trajectories

    featurizer = DihedralFeaturizer(sincos=True)
    features = featurizer.transform(trajectories[0:1])
    features = [features[0][::10]]

    tica = tICA(lag_time=1, n_components=2)
    ktica = KernelTICA(lag_time=1, kernel='linear', n_components=2,
                       random_state=42)

    tica_out = tica.fit_transform(features)[0]
    ktica_out = ktica.fit_transform(features)[0]

    assert_array_almost_equal(ktica_out, tica_out, decimal=1)


def test_ktica_compare_to_pipeline():
    X = random.randn(100, 5)

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
