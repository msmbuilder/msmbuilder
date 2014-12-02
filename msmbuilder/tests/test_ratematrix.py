from __future__ import print_function
import time
import numpy as np
import scipy.linalg
from scipy.optimize import check_grad
from msmbuilder.msm import _ratematrix
from msmbuilder.msm import ContinuousTimeMSM, MarkovStateModel
from msmbuilder.example_datasets import load_doublewell
from msmbuilder.cluster import NDGrid
np.random.seed(0)


def dense_exptheta(n):
    return np.exp(np.random.randn(n*(n-1)/2 + n))


def sparse_exptheta(n):
    exptheta = np.exp(np.random.randn(n*(n-1)/2 + n))
    zero_out = np.random.randint(low=0, high=n*(n-1)/2, size=2)
    exptheta[zero_out] = 0

    exp_d = exptheta
    exp_sp = exptheta[np.nonzero(exptheta)]
    inds_sp = np.nonzero(exptheta)[0]
    return exp_d, exp_sp, inds_sp


def test_buildK_1():
    # test buildK in sparse mode vs. dense mode
    n = 4
    exptheta = dense_exptheta(n)
    u = np.arange(n*(n-1)/2 + n).astype(np.intp)
    K1 = np.zeros((n, n))
    K2 = np.zeros((n, n))

    _ratematrix.buildK(exptheta, n, u, K1)
    _ratematrix.buildK(exptheta, n, None, K2)
    np.testing.assert_array_equal(K1, K2)


def test_buildK_2():
    # test buildK in sparse mode vs. dense mode
    n = 4
    exp_d, exp_sp, inds_sp = sparse_exptheta(n)

    K1 = np.zeros((n, n))
    K2 = np.zeros((n, n))

    _ratematrix.buildK(exp_sp, n, inds_sp, K1)
    _ratematrix.buildK(exp_d, n, None, K2)

    np.testing.assert_array_equal(K1, K2)


def test_dK_dtheta_1():
    # test function `dK_dtheta` against the numerical gradient of `buildK`
    # using dense parameterization
    n = 4
    A = np.eye(n)
    exptheta = dense_exptheta(n)

    def g(i):
        h = 1e-7
        e = np.zeros_like(exptheta)
        e[i] = h
        K1 = np.zeros((n, n))
        K2 = np.zeros((n, n))
        _ratematrix.buildK(exptheta, n, None, K1)
        _ratematrix.buildK(np.exp(np.log(exptheta) + e), n, None, K2)
        return (K2 - K1) / h

    for u in range(len(exptheta)):
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta_A(exptheta, n, u, None, A, dKu)
        np.testing.assert_array_almost_equal(g(u), dKu)


def test_dK_dtheta_2():
    # test function `dK_dtheta` against the numerical gradient of `buildK`
    # using sparse parameterization
    n = 4
    A = np.eye(n)
    _, exp_sp, inds_sp = sparse_exptheta(n)

    def g(i):
        h = 1e-7
        e = np.zeros_like(exp_sp)
        e[i] = h
        K1 = np.zeros((n, n))
        K2 = np.zeros((n, n))
        _ratematrix.buildK(exp_sp, n, inds_sp, K1)
        _ratematrix.buildK(np.exp(np.log(exp_sp) + e), n, inds_sp, K2)
        return (K2 - K1) / h

    for u in range(len(exp_sp)):
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta_A(exp_sp, n, u, inds_sp, A, dKu)
        np.testing.assert_array_almost_equal(g(u), dKu)


def test_dk_dtheta_3():
    # test function `dK_dtheta` against the numerical gradient of `buildK`
    # using sparse parameterization, plus the matrix multiply
    n = 4
    A = np.random.randn(n, n)
    _, exp_sp, inds_sp = sparse_exptheta(n)

    def g(i):
        h = 1e-7
        e = np.zeros_like(exp_sp)
        e[i] = h
        K1 = np.zeros((n, n))
        K2 = np.zeros((n, n))
        _ratematrix.buildK(exp_sp, n, inds_sp, K1)
        _ratematrix.buildK(np.exp(np.log(exp_sp) + e), n, inds_sp, K2)
        return np.dot((K2 - K1) / h, A)

    for u in range(len(exp_sp)):
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta_A(exp_sp, n, u, inds_sp, A, dKu)
        np.testing.assert_array_almost_equal(g(u), dKu)


def test_dK_dtheta_4():
    # test the matrix multiply in dK_dtheta_A
    n = 4
    A = np.random.randn(n, n)
    eye = np.eye(4)
    exptheta = np.exp(np.random.randn(n*(n-1)/2 + n))

    for u in range(len(exptheta)):
        dKu = np.zeros((n, n))
        dKuA = np.zeros((n, n))
        _ratematrix.dK_dtheta_A(exptheta, n, u, None, eye, dKu)
        _ratematrix.dK_dtheta_A(exptheta, n, u, None, A, dKuA)
        np.testing.assert_array_almost_equal(
            np.dot(dKu, A), dKuA)


def test_grad_logl_1():
    # test the gradient of the `loglikelihood` against a numerical gradient
    n = 4
    C = np.random.randint(10, size=(n, n)).astype(float)
    theta0 = np.log(dense_exptheta(n))

    def func(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=None, t=1.1)[0]

    def grad(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=None, t=1.1)[1]

    assert check_grad(func, grad, theta0) < 1e-4


def test_grad_logl_2():
    # test the gradient of the `loglikelihood` against a numerical gradient
    # using the sparse parameterization
    n = 4
    C = np.random.randint(10, size=(n, n)).astype(float)
    _, exptheta_sp, inds_sp = sparse_exptheta(n)
    theta0 = np.log(exptheta_sp)

    def func(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=inds_sp, t=1.1)[0]

    def grad(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=inds_sp, t=1.1)[1]

    assert check_grad(func, grad, theta0) < 1e-4


def test_hessian():
    grid = NDGrid(n_bins_per_feature=10, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])
    seqs = [seqs[i] for i in range(10)]

    lag_time = 120
    model = ContinuousTimeMSM(verbose=True, lag_time=lag_time)
    model.fit(seqs)
    msm = MarkovStateModel(verbose=False, lag_time=lag_time)
    print(model.summarize())
    print('MSM timescales\n', msm.fit(seqs).timescales_)
    # print('Uncertainty K\n', model.uncertainty_K())
    print('Uncertainty pi\n', model.uncertainty_pi())


def test_fit_1():
    # call fit, compare to MSM
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = ContinuousTimeMSM(verbose=False)
    model.fit([sequence])

    msm = MarkovStateModel(verbose=False)
    msm.fit([sequence])

    # they shouldn't be equal in general, but for this input they seem to be
    np.testing.assert_array_almost_equal(model.transmat_, msm.transmat_)


def test_fit_2():
    grid = NDGrid(n_bins_per_feature=5, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinuousTimeMSM(verbose=True, lag_time=10)
    model.fit(seqs)
    t1 = np.sort(model.timescales_)
    t2 = -1/np.sort(np.log(np.linalg.eigvals(model.transmat_))[1:])

    model = MarkovStateModel(verbose=False, lag_time=10)
    model.fit(seqs)
    t3 = np.sort(model.timescales_)

    np.testing.assert_array_almost_equal(t1, t2)
    # timescales should be similar to MSM (withing 50%)
    assert abs(t1[-1] - t3[-1]) / t1[-1] < 0.50


def profile():
    n = 50
    C = np.random.randint(100, size=(n, n)).astype(np.double)
    theta = np.random.randn(n*(n-1)/2 + n)

    def run():
        for i in range(5):
            _ratematrix.loglikelihood(theta, C, n, t=1.1, n_threads=8)

    start = time.time()
    run()
    print('ratematrix profile: %s' % (time.time() - start))
