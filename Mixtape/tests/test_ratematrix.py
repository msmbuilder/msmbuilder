from __future__ import print_function
import time
import numpy as np
import scipy.linalg
from mixtape.msm import _ratematrix
from mixtape.msm import ContinousTimeMSM, MarkovStateModel
from mixtape.example_datasets import load_doublewell
from mixtape.cluster import NDGrid


def test_dK_dtheta():
    # test function `dK_dtheta` against the numerical gradient of `buildK`
    n = 4
    theta = np.random.randn(n*(n-1)/2 + n)
    exptheta = np.exp(theta)

    def g(i):
        h = 1e-7
        e = np.zeros_like(exptheta)
        e[i] = h
        K1 = np.zeros((n, n))
        K2 = np.zeros((n, n))
        _ratematrix.buildK(exptheta, n, K1)
        _ratematrix.buildK(np.exp(np.log(exptheta) + e), n, K2)
        return (K2 - K1) / h

    for u in range(len(exptheta)):
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta(exptheta, n, u, dKu)

        np.testing.assert_array_almost_equal(g(u), dKu)


def test_grad_logl():
    # test the gradient of the `loglikelihood` against a numerical gradient
    n = 4
    C = np.random.randint(10, size=(n, n)).astype(float)
    theta = np.random.randn(n*(n-1)/2 + n)
    exptheta = np.exp(theta)

    h = 1e-7
    def bump(i):
        e = np.zeros_like(exptheta)
        e[i] = h
        return e

    def objective(exptheta, t):
        K = np.zeros((n, n))
        _ratematrix.buildK(exptheta, n, K)
        T = scipy.linalg.expm(K * t)
        return np.multiply(C, np.log(T)).sum()

    def deriv(exptheta, i, t):
        o1 = objective(exptheta, t)
        o2 = objective(np.exp(np.log(exptheta) + bump(i)), t)
        return (o2 - o1) / h

    def grad(exptheta, t):
        g = np.array([deriv(exptheta, i, t) for i in range(len(exptheta))])
        return objective(exptheta, t), g

    analytic_f, analytic_grad = _ratematrix.loglikelihood(theta, C, n, t=1.1)
    numerical_f, numerical_grad = grad(exptheta, t=1.1)
    np.testing.assert_array_almost_equal(analytic_grad, numerical_grad, decimal=5)
    np.testing.assert_almost_equal(analytic_f, numerical_f)


def test_fit_1():
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = ContinousTimeMSM(verbose=False)
    model.fit([sequence])

    msm = MarkovStateModel(verbose=False)
    msm.fit([sequence])

    # they shouldn't be equal in general, but for this input they seem to be
    np.testing.assert_array_almost_equal(model.transmat_, msm.transmat_)


def test_fit_2():
    grid = NDGrid(n_bins_per_feature=5, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinousTimeMSM(verbose=False, lag_time=10)
    model.fit(seqs)
    t1 = np.sort(model.timescales_)
    t2 = -1/np.sort(np.log(np.linalg.eigvals(model.transmat_))[1:])

    model = MarkovStateModel(verbose=False, lag_time=10)
    model.fit(seqs)
    t3 = np.sort(model.timescales_)

    print(t1)
    print(t2)
    print(t3)

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
