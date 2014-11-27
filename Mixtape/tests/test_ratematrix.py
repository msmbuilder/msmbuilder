from __future__ import print_function
import time
import numpy as np
import scipy.linalg
from mixtape.msm import _ratematrix
from mixtape.msm import ContinousTimeMSM, MarkovStateModel
from mixtape.example_datasets import load_doublewell
from mixtape.cluster import NDGrid
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
    u = np.arange(n*(n-1)/2 + n)
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
        print(u)
        print(dKu)
        print()
        np.testing.assert_array_almost_equal(g(u), dKu)


def test_dK_dtheta_2():
    # test function `dK_dtheta` against the numerical gradient of `buildK`
    # using sparse parameterization
    n = 4
    A = np.eye(n)
    _, exp_sp, inds_sp = sparse_exptheta(n)
    print(exp_sp, inds_sp)

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
        # print(u, inds_sp[u])
        # print(g(u))
        # print()
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta_A(exp_sp, n, u, inds_sp, A, dKu)
        np.testing.assert_array_almost_equal(g(u), dKu)
        print('passed u', u)



def test_dK_dtheta_A_1():
    n = 4
    A = np.random.randn(n, n)
    exptheta = np.exp(np.random.randn(n*(n-1)/2 + n))

    for u in range(len(exptheta)):
        dKu = np.zeros((n, n))
        dKuA = np.zeros((n, n))
        _ratematrix.dK_dtheta(exptheta, n, u, dKu)
        _ratematrix.dK_dtheta_A(exptheta, n, u, None, A, dKuA)
        np.testing.assert_array_almost_equal(
            np.dot(dKu, A), dKuA)


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


def _test_hessian():
    grid = NDGrid(n_bins_per_feature=4, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])
    seqs = [seqs[i] for i in range(1)]

    lag_time = 120
    model = ContinousTimeMSM(verbose=True, lag_time=lag_time)
    model.fit(seqs)
    print('MSM timescales\n', MarkovStateModel(verbose=False, lag_time=lag_time).fit(seqs).timescales_)

    np.set_printoptions(precision=3)
    print(model.ratemat_)

    theta = model.optimizer_state_.x
    c = model.countsmat_
    n = c.shape[0]

    hessian = _ratematrix.hessian(theta, c, n, t=1)
    print('Hessian\n', hessian)

    information = scipy.linalg.pinv(hessian)
    # print(information)
    # print('sigmaK\n', _ratematrix.uncertainty_K(information, theta, n))
    print('sigmaPi\n', _ratematrix.uncertainty_pi(information, theta, n))


def _test_fit_1():
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = ContinousTimeMSM(verbose=False)
    model.fit([sequence])

    msm = MarkovStateModel(verbose=False)
    msm.fit([sequence])

    # they shouldn't be equal in general, but for this input they seem to be
    np.testing.assert_array_almost_equal(model.transmat_, msm.transmat_)


def _test_fit_2():
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
