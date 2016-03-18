import numpy as np
from mdtraj.testing import eq
from numpy.testing import assert_approx_equal

from msmbuilder.decomposition import tICA


def test_fit_transform():
    np.random.seed(42)
    X = np.random.randn(10, 3)

    tica = tICA(n_components=2, lag_time=1)
    y2 = tica.fit_transform([np.copy(X)])[0]


def test_singular_1():
    tica = tICA(n_components=1)

    # make some data that has one column repeated twice
    X = np.random.randn(100, 2)
    X = np.hstack((X, X[:, 0, np.newaxis]))

    tica.fit([X])
    assert tica.components_.dtype == np.float64
    assert tica.eigenvalues_.dtype == np.float64


def test_singular_2():
    tica = tICA(n_components=1)

    # make some data that has one column of all zeros
    X = np.random.randn(100, 2)
    X = np.hstack((X, np.zeros((100, 1))))

    tica.fit([X])
    assert tica.components_.dtype == np.float64
    assert tica.eigenvalues_.dtype == np.float64


def test_shape():
    model = tICA(n_components=3).fit([np.random.randn(100, 10)])
    eq(model.eigenvalues_.shape, (3,))
    eq(model.eigenvectors_.shape, (10, 3))
    eq(model.components_.shape, (3, 10))


def test_score_1():
    X = np.random.randn(100, 5)
    for n in range(1, 5):
        tica = tICA(n_components=n, shrinkage=0)
        tica.fit([X])
        assert_approx_equal(
                tica.score([X]),
                tica.eigenvalues_.sum())
        X2 = np.random.randn(100, 5)
        assert tica.score([X2]) < tica.score([X])
        assert_approx_equal(tica.score([X]), tica.score_)


def test_score_2():
    X = np.random.randn(100, 5)
    Y = np.random.randn(100, 5)
    model = tICA(shrinkage=0.0, n_components=2).fit([X])
    s1 = model.score([Y])
    s2 = tICA(shrinkage=0.0).fit(model.transform([Y])).eigenvalues_.sum()

    eq(s1, s2)


def test_multiple_components():
    X = np.random.randn(100, 5)
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


def test_kinetic_mapping():
    np.random.seed(42)
    X = np.random.randn(10, 3)

    tica1 = tICA(n_components=2, lag_time=1)
    tica2 = tICA(n_components=2, lag_time=1, kinetic_mapping=True)

    y1 = tica1.fit_transform([np.copy(X)])[0]
    y2 = tica2.fit_transform([np.copy(X)])[0]

    assert eq(y2, y1*tica1.eigenvalues_)
