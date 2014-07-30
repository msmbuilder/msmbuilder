from __future__ import division
import scipy.optimize
import scipy.misc
import numpy as np
from mixtape import _reversibility

def test_logsymsumexp():  
    x = (1+np.ones(9)).reshape(3,3)
    x = x + x.T
    xx = np.zeros(3*4/2)
    k = 0
    for i in range(3):
        for j in range(i, 3):
            xx[k] = x[i,j]
            k += 1

    r1 = _reversibility.logsymsumexp(xx, 3)
    r2 = scipy.misc.logsumexp(x, axis=0)
    r3 = scipy.misc.logsumexp(x, axis=1)
    np.testing.assert_array_almost_equal(r1, r2)
    np.testing.assert_array_almost_equal(r1, r3)


def test_gradients():
    n_states = 5
    u0 = np.random.rand((n_states*(n_states+1))/2)
    symcounts = np.random.rand((n_states*(n_states+1))/2)
    rowsums = np.random.rand(n_states)
    logrowsums = np.log(rowsums)
    
    error = scipy.optimize.check_grad(_reversibility.reversible_transmat_likelihood,
                    _reversibility.reversible_transmat_grad,
                    u0, symcounts, rowsums, logrowsums)
    assert error < 1e-5


def test_reversible_mle():
    import scipy.sparse.linalg

    C = 1.0*np.array([[6, 3, 7], [4, 6, 9], [2, 6, 7]])
    # generated with msmbuilder
    result = np.array([[ 0.37499995,  0.2370208,  0.38797925],
                       [ 0.16882446,  0.31578918,  0.51538636],
                       [ 0.18615565,  0.34717763,  0.46666672]])

    T, pi = _reversibility.reversible_transmat(C)

    np.testing.assert_array_almost_equal(T, result)
    u, v = scipy.sparse.linalg.eigs(T.T, k=1)
    np.testing.assert_array_almost_equal(np.real(v / v.sum()).flatten(), pi)
