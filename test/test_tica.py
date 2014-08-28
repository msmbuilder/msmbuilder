
import numpy as np
from numpy.testing import assert_approx_equal
from mdtraj.testing import eq
from mixtape.tica import tICA
from msmbuilder.reduce import tICA as tICAr

def test_1():
    # verify that mixtape.tica.tICA and another implementation
    # of the method in msmbuilder give identicial results.
    np.random.seed(42)
    X = np.random.randn(10, 3)


    ticar = tICAr(lag=1)
    ticar.train(prep_trajectory=np.copy(X))
    y1 = ticar.project(prep_trajectory=np.copy(X), which=[0, 1])

    tica = tICA(n_components=2, lag_time=1)
    y2 = tica.fit_transform([np.copy(X)])[0]

    # check all of the internals state of the two implementations
    np.testing.assert_array_almost_equal(ticar.corrs, tica._outer_0_to_T_lagged)
    np.testing.assert_array_almost_equal(ticar.sum_t, tica._sum_0_to_TminusTau)
    np.testing.assert_array_almost_equal(ticar.sum_t_dt, tica._sum_tau_to_T)
    np.testing.assert_array_almost_equal(ticar.sum_all, tica._sum_0_to_T)

    a, b = ticar.get_current_estimate()
    np.testing.assert_array_almost_equal(a, tica.offset_correlation_)
    np.testing.assert_array_almost_equal(b, tica.covariance_)

    # TODO: compare the projections. msmbuilder.reduce.tICA doesn't do
    # a mean-substaction first whereas mixtape does.


def test_singular_1():
    tica = tICA(n_components=1)

    # make some data that has one column repeated twice
    X = np.random.randn(100, 2)
    X = np.hstack((X, X[:,0, np.newaxis]))

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
    model = tICA(n_components=3).fit([np.random.randn(100,10)])
    eq(model.eigenvalues_.shape, (3,))
    eq(model.eigenvectors_.shape, (10, 3))
    eq(model.components_.shape, (3, 10))


def test_score_1():
    X = np.random.randn(100, 5)
    for n in range(1, 5):
        tica = tICA(n_components=n, gamma=0)
        tica.fit([X])
        assert_approx_equal(
            tica.score([X]),
            tica.eigenvalues_.sum())
        X2 = np.random.randn(100, 5)
        assert tica.score([X2]) < tica.score([X])
