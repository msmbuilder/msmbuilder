import numpy as np
from msmbuilder.reduce import tICA as tICA1
from mixtape.tica import tICA as tICA2

def test_1():
    # verify that mixtape.tica.tICA and another implementation
    # of the method in msmbuilder give identicial results.
    np.random.seed(42)
    X = np.random.randn(10, 3)


    tica1 = tICA1(lag=1)
    tica1.train(prep_trajectory=np.copy(X))
    y1 = tica1.project(prep_trajectory=np.copy(X), which=[0, 1])

    tica2 = tICA2(n_components=2, offset=1)
    y2 = tica2.fit_transform(np.copy(X))

    # check all of the internals state of the two implementations
    np.testing.assert_array_almost_equal(tica1.corrs, tica2._outer_0_to_T_lagged)
    np.testing.assert_array_almost_equal(tica1.sum_t, tica2._sum_0_to_TminusOffset)
    np.testing.assert_array_almost_equal(tica1.sum_t_dt, tica2._sum_tau_to_T)
    np.testing.assert_array_almost_equal(tica1.sum_all, tica2._sum_0_to_T)

    a, b = tica1.get_current_estimate()
    np.testing.assert_array_almost_equal(a, tica2.offset_correlation_)
    np.testing.assert_array_almost_equal(b, tica2.covariance_)

    # TODO: compare the projections. msmbuilder.reduce.tICA doesn't do
    # a mean-substaction first whereas mixtape does.