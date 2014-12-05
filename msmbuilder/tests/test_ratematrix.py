from __future__ import print_function
import time
import sys
import numpy as np
import scipy.linalg
from scipy.optimize import check_grad, approx_fprime
import numdifftools as nd

from msmbuilder.msm import _ratematrix
from msmbuilder.msm import ContinuousTimeMSM, MarkovStateModel
from msmbuilder.example_datasets import load_doublewell
from msmbuilder.cluster import NDGrid
random = np.random.RandomState(0)



def dense_exptheta(n):
    return np.exp(random.randn(int(n*(n-1)/2 + n)))


def sparse_exptheta(n):
    exptheta = np.exp(random.randn(int(n*(n-1)/2 + n)))
    zero_out = random.randint(low=0, high=n*(n-1)/2, size=2)
    exptheta[zero_out] = 0

    exp_d = exptheta
    exp_sp = exptheta[np.nonzero(exptheta)]
    inds_sp = np.nonzero(exptheta)[0]
    return exp_d, exp_sp, inds_sp


def test_build_ratemat_1():
    # test build_ratemat in sparse mode vs. dense mode
    n = 4
    exptheta = dense_exptheta(n)
    u = np.arange(n*(n-1)/2 + n).astype(np.intp)
    K1 = np.zeros((n, n))
    K2 = np.zeros((n, n))

    _ratematrix.build_ratemat(exptheta, n, u, K1)
    _ratematrix.build_ratemat(exptheta, n, None, K2)
    np.testing.assert_array_equal(K1, K2)


def test_build_ratemat_2():
    # test build_ratemat in sparse mode vs. dense mode
    n = 4
    exp_d, exp_sp, inds_sp = sparse_exptheta(n)

    K1 = np.zeros((n, n))
    K2 = np.zeros((n, n))

    _ratematrix.build_ratemat(exp_sp, n, inds_sp, K1)
    _ratematrix.build_ratemat(exp_d, n, None, K2)

    np.testing.assert_array_equal(K1, K2)


def test_dK_dtheta_1():
    # test function `dK_dtheta` against the numerical gradient of `build_ratemat`
    # using dense parameterization
    n = 4
    A = random.randn(4, 4)
    exptheta = dense_exptheta(n)

    def g(i):
        h = 1e-7
        e = np.zeros_like(exptheta)
        e[i] = h
        K1 = np.zeros((n, n))
        K2 = np.zeros((n, n))
        _ratematrix.build_ratemat(exptheta, n, None, K1)
        _ratematrix.build_ratemat(np.exp(np.log(exptheta) + e), n, None, K2)
        return np.sum(np.multiply(A, (K2 - K1) / h)), (K2 - K1) / h

    for u in range(len(exptheta)):
        dKu = np.zeros((n, n))
        s_dKu_A = _ratematrix.dK_dtheta_A(exptheta, n, u, None, A, dKu)
        s_ndKA_u, ndKu = g(u)
        np.testing.assert_array_almost_equal(s_ndKA_u, s_dKu_A)
        np.testing.assert_array_almost_equal(dKu, ndKu)


def test_dK_dtheta_2():
    # test function `dK_dtheta` against the numerical gradient of `build_ratemat`
    # using sparse parameterization
    n = 4
    A = random.randn(4, 4)
    _, exp_sp, inds_sp = sparse_exptheta(n)

    def g(i):
        h = 1e-7
        e = np.zeros_like(exp_sp)
        e[i] = h
        K1 = np.zeros((n, n))
        K2 = np.zeros((n, n))
        _ratematrix.build_ratemat(exp_sp, n, inds_sp, K1)
        _ratematrix.build_ratemat(np.exp(np.log(exp_sp) + e), n, inds_sp, K2)
        return np.sum(np.multiply(A, (K2 - K1) / h))

    for u in range(len(exp_sp)):
        s_dKu_A = _ratematrix.dK_dtheta_A(exp_sp, n, u, inds_sp, A)
        np.testing.assert_array_almost_equal(g(u), s_dKu_A)


def test_grad_logl_1():
    # test the gradient of the `loglikelihood` against a numerical gradient
    n = 4
    t = 1
    C = random.randint(10, size=(n, n)).astype(float)
    theta0 = np.log(dense_exptheta(n))

    def func(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=None, t=t)[0]

    def grad(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=None, t=t)[1]

    assert check_grad(func, grad, theta0) < 1e-4


def test_grad_logl_2():
    # test the gradient of the `loglikelihood` against a numerical gradient
    # using the sparse parameterization
    n = 4
    C = random.randint(10, size=(n, n)).astype(float)
    _, exptheta_sp, inds_sp = sparse_exptheta(n)
    theta0 = np.log(exptheta_sp)

    def func(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=inds_sp, t=1.1)[0]

    def grad(theta):
        return _ratematrix.loglikelihood(theta, C, n, inds=inds_sp, t=1.1)[1]

    assert check_grad(func, grad, theta0) < 1e-4


def test_dw_1():
    n = 5
    t = 1.0
    theta0 = np.log(dense_exptheta(n))

    h = 1e-7
    def bump(u):
        e = np.zeros_like(theta0)
        e[u] = h
        return e

    def grad(theta, i):
        # gradient of the ith eigenvalue of K with respect to theta
        K = np.zeros((n, n))
        _ratematrix.build_ratemat(np.exp(theta), n, None, K)
        w, V = scipy.linalg.eig(K)
        order = np.argsort(np.real(w))

        V = np.real(np.ascontiguousarray(V[:, order]))
        U = np.ascontiguousarray(scipy.linalg.inv(V).T)

        g = np.zeros(len(theta))

        for u in range(len(theta)):
            dKu = np.zeros((n, n))
            _ratematrix.dK_dtheta_A(np.exp(theta), n, u, None, None, dKu)
            out = np.zeros(n)
            temp = np.zeros(n)
            _ratematrix.dw_du(dKu, U, V, n, temp, out)
            g[u] = out[i]
        return g

    def func(theta, i):
        # ith eigenvalue of K
        K = np.zeros((n, n))
        _ratematrix.build_ratemat(np.exp(theta), n, None, K)
        w = np.real(scipy.linalg.eigvals(K))
        w = np.sort(w)
        return w[i]

    for i in range(n):
        g1 = approx_fprime(theta0, func, 1e-7, i)
        g2 = grad(theta0, i)
        assert np.linalg.norm(g1-g2) < 2e-6


def test_hessian_1():
    n = 5
    grid = NDGrid(n_bins_per_feature=n, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinuousTimeMSM(use_sparse=False).fit(seqs)
    theta = model.theta_
    C = model.countsmat_

    hessian1 = _ratematrix.hessian(theta, C, n)
    Hfun = nd.Jacobian(lambda x: _ratematrix.loglikelihood(x, C, n)[1])
    hessian2 = Hfun(theta)

    # not sure what the cutoff here should be (see plot_test_hessian)
    assert np.linalg.norm(hessian1-hessian2) < 1


def _plot_test_hessian():
    # plot the difference between the numerical hessian and the analytic
    # approximate hessian (opens Matplotlib window)
    n = 5
    grid = NDGrid(n_bins_per_feature=n, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinuousTimeMSM(use_sparse=False).fit(seqs)
    theta = model.theta_
    C = model.countsmat_

    hessian1 = _ratematrix.hessian(theta, C, n)
    Hfun = nd.Jacobian(lambda x: _ratematrix.loglikelihood(x, C, n)[1])
    hessian2 = Hfun(theta)

    import matplotlib.pyplot as pp
    pp.scatter(hessian1.flat, hessian2.flat, marker='x')
    pp.plot(pp.xlim(), pp.xlim(), 'k')
    print('Plotting...', file=sys.stderr)
    pp.show()


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
    print('Uncertainty K\n', model.uncertainty_K())
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


def test_optimize_1():
    n = 100
    grid = NDGrid(n_bins_per_feature=n, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinuousTimeMSM(use_sparse=True, verbose=True).fit(seqs)

    y, x, n = model.loglikelihoods_.T
    x = x-x[0]
    cross = np.min(np.where(n==n[-1])[0])

    #import matplotlib.pyplot as pp
    #pp.plot(x[cross], y[cross], 'kx')
    #pp.axvline(x[cross], c='k')
    #pp.plot(x, y)
    #pp.show()
