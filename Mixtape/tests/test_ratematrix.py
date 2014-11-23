from __future__ import print_function
import numpy as np
import scipy.linalg
from mixtape.msm import _ratematrix
from mixtape.msm import ContinousTimeMSM, MarkovStateModel


def test_dK_dtheta():
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


def test_grad_objective():
    n = 4
    C = np.random.randint(10, size=(n, n)).astype(float)
    theta = np.random.randn(n*(n-1)/2 + n)
    exptheta = np.exp(theta)


    h = 1e-7
    def bump(i):
        e = np.zeros_like(exptheta)
        e[i] = h
        return e

    def objective(exptheta):
        K = np.zeros((n, n))
        _ratematrix.buildK(exptheta, n, K)
        T = scipy.linalg.expm(K)
        return np.multiply(C, np.log(T)).sum()

    def deriv(exptheta, i):
        o1 = objective(exptheta)
        o2 = objective(np.exp(np.log(exptheta) + bump(i)))
        return (o2 - o1) / h

    def grad(exptheta):
        g = np.array([deriv(exptheta, i) for i in range(len(exptheta))])
        return objective(exptheta), g

    analytic_f, analytic_grad = _ratematrix.loglikelihood(theta, C, n)
    numerical_f, numerical_grad = grad(exptheta)
    np.testing.assert_array_almost_equal(analytic_grad, numerical_grad, decimal=5)
    np.testing.assert_almost_equal(analytic_f, numerical_f)


def test_fit_1():
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = ContinousTimeMSM()
    model.fit([sequence])

    msm = MarkovStateModel(verbose=False)
    msm.fit([sequence])

    # they shouldn't be equal in general, but for this input they seem to be
    np.testing.assert_array_almost_equal(model.transmat_, msm.transmat_)
